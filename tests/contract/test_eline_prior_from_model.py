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


_FOREGROUND = {"ebmv_mw": 0.1, "law": "fitzpatrick99", "rv": 3.1}


def _model(ssp, nebular_lines, foreground=None):
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(_WAVE),
        resolution=_RESOLUTION,
        nebular_lines=nebular_lines,
    )
    extra = {} if foreground is None else {"foreground": foreground}
    return SEDModel.build(
        ssp_data=_fine_ssp(ssp),
        observation=Observation(spectroscopy=spectroscopy),
        redshift=Fixed(_Z),
        **extra,
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


def _hbeta_prior_ratio(ssp, foreground=None):
    """Projected H-beta line integral divided by the model-centred prior mean."""
    with jax.enable_x64(True):
        on = _model(ssp, True, foreground)
        off = _model(ssp, False, foreground)
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
    return integral / mu


def test_hbeta_mapping_matches_spectrum_line_flux(synthetic_ssp_wide):
    """The helper's mu_Hbeta equals the lines-on minus lines-off integral to 1 %."""
    ratio = _hbeta_prior_ratio(synthetic_ssp_wide)
    print(f"ratio={ratio:.6f}")
    assert ratio == pytest.approx(1.0, rel=1e-2)


def test_hbeta_mapping_matches_spectrum_line_flux_with_foreground(synthetic_ssp_wide):
    """Under a Milky Way screen the prior mean equals the screened H-beta line flux to 1 %.

    The screen is the foreground law at E(B-V) = 0.1, so the projected line is attenuated
    at its observed wavelength and the prior mean has to carry the same factor.
    """
    ratio = _hbeta_prior_ratio(synthetic_ssp_wide, _FOREGROUND)
    print(f"ratio={ratio:.6f}")
    assert ratio == pytest.approx(1.0, rel=1e-2)


def test_model_prior_without_screen_is_unchanged():
    """With no screen the transmission argument is one: None and ones agree bit for bit."""
    index, matched = match_model_lines(_CAT, _CAT)
    args = (jnp.asarray(_LOG_L0), index, matched, _CAT, 0.1, _DL_CM, 0.8, 0.3)
    plain = model_line_prior(*args)
    ones = model_line_prior(*args, transmission=jnp.ones(_CAT.size))
    np.testing.assert_array_equal(np.asarray(plain.prior_mean), np.asarray(ones.prior_mean))
    np.testing.assert_array_equal(np.asarray(plain.prior_width), np.asarray(ones.prior_width))


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


# ── Batched path, calibration joint and the declared aperture (#3138) ─────────

_BW = np.linspace(4000.0, 7500.0, 400)
_BR = 1000.0
_BSIG_KMS = 120.0
_BLINES = ("Hbeta", "OIII_5007", "Halpha")
_BWIDTH = 0.3
_BZ = (0.05, 0.2)
_BPARAMS = ({"sfh_delayed_log_total_mass": 10.0}, {"sfh_delayed_log_total_mass": 10.2})


def _catalog(absent=None):
    """The test catalog; ``absent`` moves one line to a wavelength the model does not predict."""
    import dataclasses

    from tengri.observation.line_list import LineList

    catalog = LineList.from_names(list(_BLINES))
    if absent is None:
        return catalog
    wavelengths = np.asarray(catalog.wavelengths).copy()
    wavelengths[catalog.names.index(absent)] = 5577.34
    return dataclasses.replace(catalog, wavelengths=jnp.asarray(wavelengths))


def _btemplate(ssp, *, free_ap=False, catalog=None, foreground=None):
    """Cue template with marginalized lines under the model-centred prior, free redshift."""
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(_BW),
        resolution=_BR,
        resample="conserving",
        eline_mode="marginalized",
        eline_catalog=_catalog() if catalog is None else catalog,
        eline_prior_type="model",
        eline_prior_width_dex=_BWIDTH,
        nebular_lines=False,
    )
    extra = {"eline_aperture_frac": Uniform(0.2, 3.0)} if free_ap else {}
    if foreground is not None:
        extra["foreground"] = foreground
    model = SEDModel.build(
        ssp_data=_fine_ssp(ssp),
        observation=Observation(spectroscopy=spectroscopy),
        redshift=Uniform(0.0, 0.5),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "publish_line_fraction": True},
        **extra,
    )
    spec = model.spec.with_fixed_value("eline_sigma_kms", float(_BSIG_KMS))
    return SEDModel(spec, model.ssp_data, observation=model.observation)


def _per_galaxy_pieces(template, z, params):
    """Prediction dict, design matrix, independent wavelengths and match map at redshift ``z``."""
    from tengri.observation.eline_marginalization import (
        apply_doublet_constraints,
        model_line_match,
    )

    catalog = template.observation.spectroscopy.effective_catalog
    full = {**params, "redshift": z}
    prediction = {
        "spec_fnu": template.predict_spectrum(full),
        "log_line_lums_attenuated": template.predict_state(full).derived[
            "log_line_lums_attenuated"
        ],
    }
    design = apply_doublet_constraints(
        build_eline_design_matrix(
            jnp.asarray(_BW), jnp.asarray(catalog.wavelengths), _BR, z, _BSIG_KMS, 0.0
        ),
        catalog.build_constraint_matrix(),
    )
    independent = np.asarray(catalog.independent_wavelengths)
    index, matched = model_line_match(template, independent)
    return prediction, design, independent, index, matched


def _galaxy_data(template, z, params, seed):
    """Continuum plus lines at 1.3 times the model prior mean, with 50 % noise."""
    prediction, design, independent, index, matched = _per_galaxy_pieces(template, z, params)
    prior = model_line_prior(
        prediction["log_line_lums_attenuated"],
        index,
        matched,
        independent,
        z,
        float(luminosity_distance(z)),
        params.get("eline_aperture_frac", 1.0),
        _BWIDTH,
    )
    continuum = np.asarray(prediction["spec_fnu"])
    sigma = np.full(_BW.size, 0.5 * np.median(continuum))
    rng = np.random.default_rng(seed)
    lines = np.asarray(design) @ (1.3 * np.asarray(prior.prior_mean))
    return continuum + lines + sigma * rng.standard_normal(_BW.size), sigma


def _prior_inputs(z):
    return lambda params: (z, luminosity_distance(z), params.get("eline_aperture_frac", 1.0))


def _lines_only_class(template, z, flux, sigma, params):
    from tengri.inference.likelihoods.marginalized import ModelELineMarginalizedLikelihood

    _, design, independent, index, matched = _per_galaxy_pieces(template, z, params)
    return ModelELineMarginalizedLikelihood(
        fnu_obs=jnp.asarray(flux),
        fnu_err=jnp.asarray(sigma),
        design_matrix_builder=lambda params: design,
        prior_inputs_builder=_prior_inputs(z),
        match_index=index,
        matched=matched,
        line_wavelengths=independent,
        prior_width_dex=_BWIDTH,
    )


def _joint_class(template, z, flux, sigma, params):
    from tengri.inference.likelihoods.marginalized import CalibrationELineMarginalizedLikelihood

    _, design, independent, index, matched = _per_galaxy_pieces(template, z, params)
    return CalibrationELineMarginalizedLikelihood(
        fnu_obs=jnp.asarray(flux),
        fnu_err=jnp.asarray(sigma),
        wavelength=jnp.asarray(_BW),
        design_matrix_builder=lambda params: design,
        n_poly=2,
        prior_sigma=1.0,
        eline_prior_type="model",
        eline_line_wavelengths=independent,
        eline_prior_width_dex=_BWIDTH,
        prior_inputs_builder=_prior_inputs(z),
        match_index=index,
        matched=matched,
    )


def _batch(template, flux_sigma, zs=_BZ, ebmvs=None):
    from tengri.observation.batched import GalaxySpectrum, build_spectro_batches

    ebmvs = (None,) * len(zs) if ebmvs is None else ebmvs
    galaxies = [
        GalaxySpectrum(
            wave=_BW,
            flux=f,
            ivar=1.0 / s**2,
            z=z,
            resolution=_BR,
            **({} if e is None else {"ebmv_mw": e}),
        )
        for z, (f, s), e in zip(zs, flux_sigma, ebmvs, strict=True)
    ]
    spec, batch, _ = build_spectro_batches(galaxies, model=template)[0]
    return spec, batch


def _batched_vs_per_galaxy(template, joint, params_list):
    from tengri.inference.batched_spectro import batched_log_likelihood

    data = [_galaxy_data(template, z, p, seed=i) for i, (z, p) in enumerate(zip(_BZ, params_list))]
    spec, batch = _batch(template, data)
    kwargs = (
        {"calibration_marginalize": True, "cal_n_poly": 2, "cal_prior_sigma": 1.0} if joint else {}
    )
    loglik = jax.vmap(batched_log_likelihood(template, spec, **kwargs), in_axes=(0, 0))
    stacked = {k: jnp.asarray([p[k] for p in params_list]) for k in params_list[0]}
    got = np.asarray(loglik(stacked, batch))
    build = _joint_class if joint else _lines_only_class
    expected = []
    for z, p, (flux, sigma) in zip(_BZ, params_list, data):
        prediction, *_ = _per_galaxy_pieces(template, z, p)
        expected.append(float(build(template, z, flux, sigma, p).log_prob(prediction, p)))
    return got, np.asarray(expected), batch


@pytest.mark.parametrize("joint", [False, True])
def test_batched_equals_per_galaxy_with_model_prior(synthetic_ssp_wide, joint):
    """Two galaxies at different z (one with H-alpha off the grid): batched == per-galaxy."""
    with jax.enable_x64(True):
        template = _btemplate(synthetic_ssp_wide, free_ap=True)
        params = [
            {**_BPARAMS[0], "eline_aperture_frac": 0.8},
            {**_BPARAMS[1], "eline_aperture_frac": 1.4},
        ]
        got, expected, batch = _batched_vs_per_galaxy(template, joint, params)
    lm = np.asarray(batch.line_mask)
    assert not np.array_equal(lm[0], lm[1])
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)


def _screen(ebmv):
    return {"ebmv_mw": ebmv, "law": "fitzpatrick99", "rv": 3.1}


def _foreground_batched_vs_per_galaxy(ssp, joint, ebmvs):
    """Batched log-likelihood and per-galaxy references, each galaxy with its own screen."""
    from tengri.inference.batched_spectro import batched_log_likelihood

    templates = [_btemplate(ssp, foreground=_screen(e)) for e in ebmvs]
    data = [
        _galaxy_data(t, z, p, seed=i) for i, (t, z, p) in enumerate(zip(templates, _BZ, _BPARAMS))
    ]
    batch_template = _btemplate(ssp, foreground=_screen(0.0))
    spec, batch = _batch(batch_template, data, ebmvs=ebmvs)
    kwargs = (
        {"calibration_marginalize": True, "cal_n_poly": 2, "cal_prior_sigma": 1.0} if joint else {}
    )
    loglik = jax.vmap(batched_log_likelihood(batch_template, spec, **kwargs), in_axes=(0, 0))
    stacked = {k: jnp.asarray([p[k] for p in _BPARAMS]) for k in _BPARAMS[0]}
    got = np.asarray(loglik(stacked, batch))
    build = _joint_class if joint else _lines_only_class
    expected = []
    for t, z, p, (flux, sigma) in zip(templates, _BZ, _BPARAMS, data):
        prediction, *_ = _per_galaxy_pieces(t, z, p)
        expected.append(float(build(t, z, flux, sigma, p).log_prob(prediction, p)))
    return got, np.asarray(expected)


@pytest.mark.parametrize("joint", [False, True])
def test_batched_equals_per_galaxy_with_model_prior_and_foreground(synthetic_ssp_wide, joint):
    """Two galaxies, each with its own Milky Way E(B-V): batched == per-galaxy, both paths."""
    with jax.enable_x64(True):
        got, expected = _foreground_batched_vs_per_galaxy(synthetic_ssp_wide, joint, (0.05, 0.3))
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)


def test_batched_unmatched_catalog_line_is_flat_on_both_paths(synthetic_ssp_wide):
    """A catalog line the model does not predict is flat: batched == per-galaxy, joint too."""
    from tengri.observation.eline_marginalization import model_line_match

    with jax.enable_x64(True):
        template = _btemplate(synthetic_ssp_wide, catalog=_catalog(absent="OIII_5007"))
        catalog = template.observation.spectroscopy.effective_catalog
        _, matched = model_line_match(template, np.asarray(catalog.independent_wavelengths))
        assert list(matched) == [True, False, True]
        for joint in (False, True):
            got, expected, _ = _batched_vs_per_galaxy(template, joint, list(_BPARAMS))
            np.testing.assert_allclose(got, expected, rtol=1e-10, atol=0.0)


def test_flat_column_ignores_the_model_line_it_points_to():
    """Class level: an unmatched column's value does not move with the luminosity at its index."""
    sigma = np.full(_N_PIX, 1.0e-25)
    matched = np.array([True, False])
    design, _ = _class_setup(np.zeros(_N_PIX), sigma, matched)
    data = np.asarray(_prediction(4.0)["spec_fnu"]) + design @ np.array([1.0e-25, 5.0e-26])
    _, lk = _class_setup(data, sigma, matched)
    base = _prediction(4.0)
    lums = base["log_line_lums_attenuated"]
    moved_flat = {**base, "log_line_lums_attenuated": lums.at[1].add(1.5)}
    moved_gaussian = {**base, "log_line_lums_attenuated": lums.at[0].add(1.5)}
    assert float(lk.log_prob(base, {})) == float(lk.log_prob(moved_flat, {}))
    assert float(lk.log_prob(base, {})) != float(lk.log_prob(moved_gaussian, {}))


def test_one_compile_per_spec_with_model_prior(synthetic_ssp_wide):
    """A second batch of the same spec (other z, other line masks) triggers no compile."""
    from jax._src import test_util as jtu

    from tengri.inference.batched_spectro import batched_log_likelihood

    with jax.enable_x64(True):
        template = _btemplate(synthetic_ssp_wide)
        data = [_galaxy_data(template, _BZ[0], _BPARAMS[0], seed=i) for i in range(2)]
        spec_a, batch_a = _batch(template, data, _BZ)
        spec_b, batch_b = _batch(template, data, (0.2, 0.08))
        assert spec_a == spec_b
        assert not np.array_equal(np.asarray(batch_a.line_mask), np.asarray(batch_b.line_mask))
        loglik = jax.jit(jax.vmap(batched_log_likelihood(template, spec_a), in_axes=(0, 0)))
        params = {"sfh_delayed_log_total_mass": jnp.full((2,), 10.0)}
        jax.block_until_ready(loglik(params, batch_a))
        with jtu.count_jit_compilation_cache_miss() as counter:
            jax.block_until_ready(loglik(params, batch_b))
    assert counter() == 0


def test_batched_float32_gradients_wrt_aperture_and_mass(synthetic_ssp_wide):
    """Float32 at log10 M = 4.0 with scaled data: finite value, finite nonzero gradients."""
    from tengri.inference.batched_spectro import batched_log_likelihood

    scale = 1.0e-6
    with jax.enable_x64(False):
        template = _btemplate(synthetic_ssp_wide, free_ap=True)
        rng = np.random.default_rng(0)
        flux = rng.normal(1.3e-15, 0.1e-15, _BW.size) * scale
        sigma = np.full(_BW.size, 0.1e-15 * scale)
        spec, batch = _batch(template, [(flux, sigma), (flux, sigma)])
        loglik = batched_log_likelihood(template, spec)
        obs = batch.galaxy(0)
        params = {"sfh_delayed_log_total_mass": 4.0, "eline_aperture_frac": 0.9}
        value = float(loglik(params, obs))
        grad = jax.grad(lambda p: loglik(p, obs))(params)
        g_mass = float(grad["sfh_delayed_log_total_mass"])
        g_ap = float(grad["eline_aperture_frac"])
    assert np.isfinite(value)
    assert np.isfinite(g_mass)
    assert g_mass != 0.0
    assert np.isfinite(g_ap)
    assert g_ap != 0.0


def test_aperture_round_trips_through_the_grammar_and_both_paths(synthetic_ssp_wide):
    """A freed aperture survives to_groups -> build, and is a latent on the batched path."""
    from tengri.inference.batched_spectro import _free_latent_names

    with jax.enable_x64(True):
        template = _btemplate(synthetic_ssp_wide, free_ap=True)
        rebuilt = SEDModel.build(
            ssp_data=template.ssp_data,
            observation=template.observation,
            **template.spec.to_groups(),
        )
        pinned = _btemplate(synthetic_ssp_wide)
    assert "eline_aperture_frac" in rebuilt.spec.free_params
    assert rebuilt.spec.free_params == template.spec.free_params
    assert "eline_aperture_frac" in _free_latent_names(template)
    assert "eline_aperture_frac" not in pinned.spec.free_params
    assert pinned.spec.get_fixed_values()["eline_aperture_frac"] == 1.0


def test_one_forward_serves_the_spectrum_and_the_line_luminosities(synthetic_ssp_wide):
    """The threaded kernel returns the spectrum and the attenuated lines of one forward."""
    with jax.enable_x64(True):
        template = _btemplate(synthetic_ssp_wide)
        params = {**_BPARAMS[0], "redshift": _BZ[0]}
        kernel = template._get_or_build_predict_observables_lines_jit()
        threaded = template._resolve_threaded_data(None, None, None)
        observables, lums = kernel(params, template.spec.get_fixed_values(), *threaded)
        reference = template.predict_state(params).derived["log_line_lums_attenuated"]
        spectrum = template.predict_spectrum(params)
    np.testing.assert_allclose(np.asarray(lums), np.asarray(reference), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(observables.spec_fnu), np.asarray(spectrum), rtol=1e-10)
    assert float(jnp.max(lums)) > 30.0
