# SPDX-License-Identifier: BSD-3-Clause
"""Emission-line marginalization: the wrapper over the linear-Gaussian marginal (#3117)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.eline_marginalization import marginalize_emission_lines

pytestmark = pytest.mark.contract


def _toy(n_pix: int = 40, seed: int = 0):
    """Three Gaussian line columns on a noisy residual, with heteroscedastic noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_pix)
    centers = np.array([-0.5, 0.0, 0.4])
    design = np.exp(-0.5 * ((x[:, None] - centers[None, :]) / 0.05) ** 2)
    sigma = 0.1 + 0.05 * rng.random(n_pix)
    residual = design @ np.array([0.7, -0.3, 1.1]) + sigma * rng.standard_normal(n_pix)
    return residual, sigma, design


def _dense_marginal(residual, sigma, design, prior_variance):
    """Marginal log-likelihood from the dense covariance diag(sigma^2) + G Lambda G^T.

    The normalization is the one of the wrapper: the constant -1/2 N ln 2pi and the
    -sum ln sigma of the full Gaussian are removed, so the value is -1/2 ln det C +
    sum ln sigma - 1/2 r^T C^-1 r.
    """
    cov = np.diag(sigma**2) + design @ np.diag(prior_variance) @ design.T
    _sign, logdet = np.linalg.slogdet(cov)
    quad = residual @ np.linalg.solve(cov, residual)
    return -0.5 * logdet - 0.5 * quad + np.sum(np.log(sigma))


def _old_closed_form(residual, noise, design, prior_variance):
    """Frozen copy of the pre-#3117 closed form, kept as the reference for the fix."""
    gw = design / noise[:, None]
    rw = residual / noise
    gtg = gw.T @ gw
    gtr = gw.T @ rw
    cov = np.linalg.inv(gtg + np.diag(1.0 / prior_variance))
    a_hat = cov @ gtr
    chi2_marg = rw @ rw - a_hat @ gtg @ a_hat
    prior_penalty = np.sum(a_hat**2 / prior_variance)
    _sign, logdet_cov = np.linalg.slogdet(cov)
    log_det_corr = logdet_cov - np.sum(np.log(prior_variance))
    return -0.5 * chi2_marg - 0.5 * prior_penalty + 0.5 * log_det_corr, a_hat


def test_value_is_the_exact_marginal():
    residual, sigma, design = _toy()
    prior_variance = np.full(3, 0.5)
    ln_l, a_hat, _a_cov = marginalize_emission_lines(
        residual, sigma, design, prior_variance=prior_variance
    )
    expected = _dense_marginal(residual, sigma, design, prior_variance)
    assert np.isclose(float(ln_l), expected, rtol=1e-10, atol=0.0)
    assert np.all(np.isfinite(np.asarray(a_hat)))


def test_fix_is_exactly_the_prior_penalty_the_old_formula_double_counted():
    residual, sigma, design = _toy()
    prior_variance = np.full(3, 0.5)
    ln_l, _a_hat, _a_cov = marginalize_emission_lines(
        residual, sigma, design, prior_variance=prior_variance
    )
    old_value, a_old = _old_closed_form(residual, sigma, design, prior_variance)
    penalty = np.sum(a_old**2 / prior_variance)
    assert np.isclose(float(ln_l) - old_value, penalty, rtol=1e-10, atol=0.0)


def test_presence_mask_drops_masked_pixels_exactly():
    """A masked pixel is absent from every term: the value equals the class run without it."""
    import jax.numpy as jnp

    from tengri.inference.likelihoods.marginalized import ELineMarginalizedLikelihood

    residual, sigma, design = _toy()
    prior_variance = np.full(3, 0.5)
    mask = np.ones_like(residual)
    mask[5] = 0.0
    keep = mask > 0
    masked_lk = ELineMarginalizedLikelihood(
        fnu_obs=jnp.asarray(residual),
        fnu_err=jnp.asarray(sigma),
        design_matrix=jnp.asarray(design),
        prior_variance=jnp.asarray(prior_variance),
        presence_key="presence",
    )
    actual = float(
        masked_lk.log_prob(
            {"spec_fnu": jnp.zeros_like(jnp.asarray(residual))},
            data_args={"presence": jnp.asarray(mask)},
        )
    )
    dropped_lk = ELineMarginalizedLikelihood(
        fnu_obs=jnp.asarray(residual[keep]),
        fnu_err=jnp.asarray(sigma[keep]),
        design_matrix=jnp.asarray(design[keep]),
        prior_variance=jnp.asarray(prior_variance),
    )
    expected = float(dropped_lk.log_prob({"spec_fnu": jnp.zeros(int(keep.sum()))}))
    assert np.isclose(actual, expected, rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("prior_sigma", [0.5, 5.0])
def test_lines_only_prior_width_is_the_eline_prior_sigma(prior_sigma):
    """The line prior variance is the knob: the value equals the dense marginal at that width."""
    import jax.numpy as jnp

    from tengri.inference.likelihoods.marginalized import ELineMarginalizedLikelihood

    residual, sigma, design = _toy()
    lk = ELineMarginalizedLikelihood(
        fnu_obs=jnp.asarray(residual),
        fnu_err=jnp.asarray(sigma),
        design_matrix=jnp.asarray(design),
        prior_variance=prior_sigma**2,
    )
    actual = float(lk.log_prob({"spec_fnu": jnp.zeros_like(jnp.asarray(residual))}))
    expected = _dense_marginal(residual, sigma, design, np.full(3, prior_sigma**2))
    assert np.isclose(actual, expected, rtol=1e-10, atol=0.0)


# ── Batched path with eline_mode="marginalized" (#3117) ──────────────────────

_MARG_LINES = ("Hbeta", "OIII_5007", "Halpha")
_MARG_R = 300.0
_MARG_SIGMA_KMS = 120.0
_MARG_PRIOR_SIGMA = 1.0e-15
_MARG_WAVE = np.linspace(4000.0, 7500.0, 160)


def _marg_template(ssp, *, eline_mode="marginalized", lines=_MARG_LINES, prior_type="flat"):
    """Free-redshift template fitting ``lines``; the stellar mass is the only physical latent."""
    from tengri import Fixed, Observation, SEDModel, Spectroscopy, Uniform
    from tengri.observation.line_list import LineList

    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(_MARG_WAVE),
        resolution=_MARG_R,
        resample="conserving",
        eline_mode=eline_mode,
        eline_catalog=LineList.from_names(list(lines)),
        eline_prior_sigma=_MARG_PRIOR_SIGMA,
        eline_prior_type=prior_type,
        eline_broad=False,
    )
    from tengri import DEFAULT

    model = SEDModel.build(
        ssp_data=ssp,
        observation=Observation(spectroscopy=spectroscopy),
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "sfh_dpl_log_total_mass": Uniform(8.0, 12.0),
        },
        neb={"type": "none"},
        redshift=Uniform(0.0, 0.5),
    )
    spec = model.spec.with_fixed_value("eline_sigma_kms", float(_MARG_SIGMA_KMS))
    return SEDModel(spec, model.ssp_data, observation=model.observation)


def _marg_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    flux = rng.normal(1.3e-15, 0.1e-15, _MARG_WAVE.size)
    sigma = np.full(_MARG_WAVE.size, 0.1e-15)
    return flux, sigma


def _marg_line_design(model, z):
    from tengri.observation.eline_marginalization import (
        apply_doublet_constraints,
        build_eline_design_matrix,
    )

    catalog = model.observation.spectroscopy.effective_catalog
    g = build_eline_design_matrix(
        jnp.asarray(_MARG_WAVE),
        jnp.asarray(catalog.wavelengths),
        _MARG_R,
        z,
        _MARG_SIGMA_KMS,
        0.0,
    )
    return apply_doublet_constraints(g, catalog.build_constraint_matrix())


def test_batched_marginalized_lines_match_per_galaxy_lines_only(synthetic_ssp):
    """Single galaxy: the batched lines-only marginal equals ELineMarginalizedLikelihood."""
    from tengri.forward.batched_spectrum import check_template_model  # noqa: F401
    from tengri.inference.batched_spectro import batched_log_likelihood
    from tengri.inference.likelihoods.marginalized import ELineMarginalizedLikelihood
    from tengri.observation.batched import GalaxySpectrum, build_spectro_batches

    z = 0.1
    template = _marg_template(synthetic_ssp)
    flux, sigma = _marg_data()
    gal = GalaxySpectrum(wave=_MARG_WAVE, flux=flux, ivar=1.0 / sigma**2, z=z, resolution=_MARG_R)
    spec, batch, _ = build_spectro_batches([gal], model=template)[0]
    params = {"sfh_dpl_log_total_mass": 10.0}
    got = float(batched_log_likelihood(template, spec)(params, batch.galaxy(0)))

    m_single = template.predict_spectrum({**params, "redshift": z})
    ref = ELineMarginalizedLikelihood(
        fnu_obs=jnp.asarray(flux),
        fnu_err=jnp.asarray(sigma),
        design_matrix=_marg_line_design(template, z),
        prior_variance=_MARG_PRIOR_SIGMA**2,
    )
    expected = float(ref.log_prob({"spec_fnu": m_single}))
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_batched_marginalized_joint_matches_per_galaxy_calibration_class(synthetic_ssp):
    """Single galaxy: the batched joint calibration+line marginal equals the per-galaxy class."""
    from tengri.inference.batched_spectro import batched_log_likelihood
    from tengri.inference.likelihoods.marginalized import (
        CalibrationELineMarginalizedLikelihood,
    )
    from tengri.observation.batched import GalaxySpectrum, build_spectro_batches

    z = 0.1
    template = _marg_template(synthetic_ssp)
    flux, sigma = _marg_data()
    gal = GalaxySpectrum(wave=_MARG_WAVE, flux=flux, ivar=1.0 / sigma**2, z=z, resolution=_MARG_R)
    spec, batch, _ = build_spectro_batches([gal], model=template)[0]
    params = {"sfh_dpl_log_total_mass": 10.0}
    loglik = batched_log_likelihood(
        template, spec, calibration_marginalize=True, cal_n_poly=2, cal_prior_sigma=1.0
    )
    got = float(loglik(params, batch.galaxy(0)))

    m_single = template.predict_spectrum({**params, "redshift": z})
    ref = CalibrationELineMarginalizedLikelihood(
        fnu_obs=jnp.asarray(flux),
        fnu_err=jnp.asarray(sigma),
        wavelength=jnp.asarray(_MARG_WAVE),
        design_matrix_builder=lambda _p: _marg_line_design(template, z),
        n_poly=2,
        prior_sigma=1.0,
        eline_prior_type="flat",
        eline_prior_sigma=_MARG_PRIOR_SIGMA,
        channel="spec_fnu",
    )
    expected = float(ref.log_prob({"spec_fnu": m_single}, params={}))
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_batched_marginalized_is_refused_only_for_cloudy_priors(synthetic_ssp):
    """The batched path accepts the flat marginal and names the Cloudy alternative."""
    from tengri.forward.batched_spectrum import check_template_model

    check_template_model(_marg_template(synthetic_ssp))
    with pytest.raises(ValueError, match="cloudy"):
        check_template_model(_marg_template(synthetic_ssp, prior_type="cloudy"))
