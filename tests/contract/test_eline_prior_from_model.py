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


_DL_CM = 1.0e26
_CAT = np.array([4900.0, 5100.0])
_LOG_L0 = np.log10(np.array([3.0e40, 1.5e40]))
_N_PIX = 60


def _class_setup(data, sigma, matched_mask, dl_cm=_DL_CM):
    """Class-level model-prior likelihood on the toy grid, with a given catalog match."""
    from tengri.inference.likelihoods.marginalized import ModelELineMarginalizedLikelihood

    waves = np.linspace(4800.0, 5400.0, _N_PIX)
    design = np.asarray(
        build_eline_design_matrix(jnp.asarray(waves), jnp.asarray(_CAT), 500.0, 0.0)
    )
    index, _ = match_model_lines(_CAT, _CAT)
    return design, ModelELineMarginalizedLikelihood(
        fnu_obs=jnp.asarray(data),
        fnu_err=jnp.asarray(sigma),
        design_matrix_builder=lambda params: jnp.asarray(design),
        prior_inputs_builder=lambda params: (0.0, dl_cm, params.get("ap", 1.0)),
        match_index=index,
        matched=np.asarray(matched_mask),
        line_wavelengths=_CAT,
        prior_width_dex=5.0,
        channel="spec_fnu",
    )


def _prediction(theta):
    """Continuum and attenuated line luminosities, both scaled by 10**(theta - 4) dex."""
    cont = jnp.full(_N_PIX, 1.0e-25) * 10.0 ** (theta - 4.0)
    return {"spec_fnu": cont, "log_line_lums_attenuated": jnp.asarray(_LOG_L0 + theta)}


def test_three_point_differences_track_flat_up_to_log_width():
    """Model-minus-flat differences over +-0.1 dex equal -sum_l d ln W_l, to rtol 1e-6."""
    sigma = np.full(_N_PIX, 1.0e-25)
    index, matched = match_model_lines(_CAT, _CAT)
    mu_true = np.asarray(
        model_line_prior(
            jnp.asarray(_LOG_L0 + 4.0), index, matched, _CAT, 0.0, _DL_CM, 1.0, 5.0
        ).prior_mean
    )
    design, _ = _class_setup(np.zeros(_N_PIX), sigma, matched)
    data = np.asarray(_prediction(4.0)["spec_fnu"]) + design @ mu_true
    _, model_lk = _class_setup(data, sigma, matched)
    _, flat_lk = _class_setup(data, sigma, np.zeros(2, dtype=bool))

    def log_pair(theta):
        pred = _prediction(theta)
        return float(model_lk.log_prob(pred, {})), float(flat_lk.log_prob(pred, {}))

    lo, mid, hi = log_pair(3.9), log_pair(4.0), log_pair(4.1)
    d_model = hi[0] - lo[0]
    d_flat = hi[1] - lo[1]
    expected = -2.0 * 0.2 * np.log(10.0)
    np.testing.assert_allclose(d_model - d_flat, expected, rtol=1e-6, atol=1e-8)
    assert np.isfinite(mid[0])


def test_narrow_width_recovers_model_amplitudes_and_beats_flat():
    """With 0.05 dex the amplitudes of model-consistent lines come back within 2 %."""
    import dataclasses

    from tengri.observation.eline_marginalization import unit_model_line_prior
    from tengri.observation.linear_marginal import marginalize_linear_nuisance

    sigma = np.full(_N_PIX, 1.0e-25)
    index, matched = match_model_lines(_CAT, _CAT)
    mu = np.asarray(
        model_line_prior(
            jnp.asarray(_LOG_L0 + 4.0), index, matched, _CAT, 0.0, _DL_CM, 1.0, 0.05
        ).prior_mean
    )
    design, _ = _class_setup(np.zeros(_N_PIX), sigma, matched)
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 1e-27, _N_PIX)
    data = np.asarray(_prediction(4.0)["spec_fnu"]) + design @ mu + noise
    prior = model_line_prior(
        jnp.asarray(_LOG_L0 + 4.0), index, matched, _CAT, 0.0, _DL_CM, 1.0, 0.05
    )
    unit = unit_model_line_prior(jnp.asarray(design), prior, noise=jnp.asarray(sigma))
    result = marginalize_linear_nuisance(
        jnp.asarray(data),
        jnp.asarray(_prediction(4.0)["spec_fnu"]),
        unit.design,
        sigma=jnp.asarray(sigma),
        prior_mean=unit.prior_mean,
        prior_precision=unit.prior_precision,
        scale=unit.scale,
    )
    amplitudes = np.asarray(prior.prior_width * result.coef_hat)
    np.testing.assert_allclose(amplitudes, mu, rtol=2e-2)
    _, narrow = _class_setup(data, sigma, matched)
    narrow = dataclasses.replace(narrow, prior_width_dex=0.05)
    _, flat = _class_setup(data, sigma, np.zeros(2, dtype=bool))
    pred = _prediction(4.0)
    assert float(narrow.log_prob(pred, {})) > float(flat.log_prob(pred, {}))


def test_class_aperture_gradient_is_finite_and_nonzero():
    """d(log-likelihood)/d(aperture_frac) through the class is finite and nonzero."""
    sigma = np.full(_N_PIX, 1.0e-25)
    _, matched = match_model_lines(_CAT, _CAT)
    design, lk = _class_setup(np.zeros(_N_PIX), sigma, matched)
    data = np.asarray(_prediction(4.0)["spec_fnu"]) + design @ np.array([1.0e-25, 5.0e-26])
    _, lk = _class_setup(data, sigma, matched)
    pred = _prediction(4.0)

    def value(ap):
        return lk.log_prob(pred, {"ap": ap})

    grad = float(jax.grad(value)(jnp.asarray(0.8)))
    assert np.isfinite(grad)
    assert grad != 0.0


def test_class_float32_scaled_data_finite_and_nonzero_gradient():
    """Float32 at scaled data: the value is finite and d/d(aperture) is finite and nonzero."""
    scale = 1.0e25
    sigma = np.full(_N_PIX, 1.0e-25 * scale)
    _, matched = match_model_lines(_CAT, _CAT)
    design, _ = _class_setup(np.zeros(_N_PIX), sigma, matched)
    data = (
        np.asarray(_prediction(4.0)["spec_fnu"]) + design @ np.array([1.0e-25, 5.0e-26])
    ) * scale
    dl_scaled = _DL_CM / np.sqrt(scale)
    with jax.enable_x64(False):
        lk = _class_setup(data, sigma, matched, dl_cm=dl_scaled)[1]
        pred = {k: jnp.asarray(v, jnp.float32) for k, v in _prediction(4.0).items()}
        pred["spec_fnu"] = pred["spec_fnu"] * scale
        value = float(lk.log_prob(pred, {"ap": jnp.float32(0.8)}))
        grad = float(jax.grad(lambda ap: lk.log_prob(pred, {"ap": ap}))(jnp.float32(0.8)))
    assert np.isfinite(value)
    assert np.isfinite(grad)
    assert grad != 0.0
