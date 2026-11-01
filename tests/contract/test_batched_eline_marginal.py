# SPDX-License-Identifier: BSD-3-Clause
"""Emission-line marginalization: the wrapper over the linear-Gaussian marginal (#3117)."""

import jax
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


def _marg_template(
    ssp,
    *,
    eline_mode="marginalized",
    lines=_MARG_LINES,
    prior_type="flat",
    banded=False,
    prior_sigma=_MARG_PRIOR_SIGMA,
    fixed_z=None,
):
    """Free-redshift template fitting ``lines``; the stellar mass is the only physical latent."""
    from tengri import Fixed, Observation, SEDModel, Spectroscopy, Uniform
    from tengri.observation.banded import gaussian_resolution_bands
    from tengri.observation.line_list import LineList

    extra = (
        {"resolution_matrix": gaussian_resolution_bands(jnp.asarray(_MARG_WAVE), _MARG_R, 21)}
        if banded
        else {}
    )
    spectroscopy = Spectroscopy(
        wave_obs=jnp.asarray(_MARG_WAVE),
        resolution=_MARG_R,
        resample="conserving",
        **extra,
        eline_mode=eline_mode,
        eline_catalog=LineList.from_names(list(lines)),
        eline_prior_sigma=prior_sigma,
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
        redshift=Uniform(0.0, 0.5) if fixed_z is None else Fixed(fixed_z),
    )
    if eline_mode == "off":
        return model
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


def _marg_galaxy(z, seed=0, resolution=_MARG_R, flux_scale=1.0):
    from tengri.observation.batched import GalaxySpectrum

    flux, sigma = _marg_data(seed)
    return GalaxySpectrum(
        wave=_MARG_WAVE,
        flux=flux * flux_scale,
        ivar=1.0 / (sigma * flux_scale) ** 2,
        z=z,
        resolution=resolution,
    )


def test_batched_marginalized_gradient_is_finite_and_nonzero(synthetic_ssp):
    """The batched marginal is differentiable in the stellar mass and the line scale."""
    from tengri.inference.batched_spectro import batched_log_likelihood
    from tengri.observation.batched import build_spectro_batches

    template = _marg_template(synthetic_ssp)
    spec, batch, _ = build_spectro_batches([_marg_galaxy(0.1)], model=template)[0]
    loglik = batched_log_likelihood(template, spec)
    obs = batch.galaxy(0)
    params = {"sfh_dpl_log_total_mass": 10.0}
    grad = jax.grad(lambda p: loglik(p, obs))(params)
    g = float(grad["sfh_dpl_log_total_mass"])
    assert np.isfinite(g)
    assert g != 0.0


def test_masked_line_equals_per_galaxy_run_without_that_column(synthetic_ssp):
    """A line redshifted past the grid is masked: the value equals the run without its column.

    At z=0.145 H-alpha lands at 7515 A, past the 7500 A edge, so its column is zero and
    the batched marginal must equal the per-galaxy marginal on the H-beta column alone.
    """
    from tengri.inference.batched_spectro import batched_log_likelihood
    from tengri.inference.likelihoods.marginalized import ELineMarginalizedLikelihood
    from tengri.observation.batched import build_spectro_batches

    z = 0.145
    template = _marg_template(synthetic_ssp, lines=("Hbeta", "Halpha"))
    spec, batch, _ = build_spectro_batches([_marg_galaxy(z)], model=template)[0]
    params = {"sfh_dpl_log_total_mass": 10.0}
    got = float(batched_log_likelihood(template, spec)(params, batch.galaxy(0)))

    g_full = _marg_line_design(template, z)
    assert float(jnp.sum(jnp.abs(g_full[:, 1]))) > 0.0
    flux, sigma = _marg_data()
    m_single = template.predict_spectrum({**params, "redshift": z})
    ref = ELineMarginalizedLikelihood(
        fnu_obs=jnp.asarray(flux),
        fnu_err=jnp.asarray(sigma),
        design_matrix=g_full[:, :1],
        prior_variance=_MARG_PRIOR_SIGMA**2,
    )
    expected = float(ref.log_prob({"spec_fnu": m_single}))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_flat_prior_column_mask_equals_deleting_the_column():
    """Flat prior: a masked line column leaves the marginal equal to deleting that column.

    Without the mask the normal matrix is singular on a flat prior, so this is the case
    that the column mask exists for.
    """
    from tengri.observation.linear_marginal import marginalize_linear_nuisance

    rng = np.random.default_rng(4)
    n_pix = 30
    design = rng.normal(size=(n_pix, 3))
    design[:, 1] = 0.0
    data = design @ np.array([0.5, 0.0, -0.8]) + 0.3 * rng.normal(size=n_pix)
    sigma = np.full(n_pix, 0.3)
    masked = marginalize_linear_nuisance(
        jnp.asarray(data),
        jnp.zeros(n_pix),
        jnp.asarray(design),
        sigma=jnp.asarray(sigma),
        col_mask=jnp.asarray([1.0, 0.0, 1.0]),
    )
    keep = np.array([0, 2])
    deleted = marginalize_linear_nuisance(
        jnp.asarray(data),
        jnp.zeros(n_pix),
        jnp.asarray(design[:, keep]),
        sigma=jnp.asarray(sigma),
    )
    assert np.isfinite(float(masked.log_like))
    np.testing.assert_allclose(float(masked.log_like), float(deleted.log_like), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(masked.coef_hat)[keep], deleted.coef_hat, rtol=1e-12)


def test_all_lines_outside_grid_equals_no_line_likelihood(synthetic_ssp):
    """Every line masked: the marginal equals the lines-off likelihood."""
    from tengri.inference.batched_spectro import batched_log_likelihood
    from tengri.observation.batched import build_spectro_batches

    z = 0.145
    on = _marg_template(synthetic_ssp, lines=("Halpha",))
    off = _marg_template(synthetic_ssp, eline_mode="off", lines=("Halpha",))
    spec_on, batch_on, _ = build_spectro_batches([_marg_galaxy(z)], model=on)[0]
    spec_off, batch_off, _ = build_spectro_batches([_marg_galaxy(z)], model=off)[0]
    assert np.all(np.asarray(batch_on.line_mask) == 0.0)
    params = {"sfh_dpl_log_total_mass": 10.0}
    got = float(batched_log_likelihood(on, spec_on)(params, batch_on.galaxy(0)))
    ref = float(batched_log_likelihood(off, spec_off)(params, batch_off.galaxy(0)))
    np.testing.assert_allclose(got, ref, rtol=1e-10)


def test_one_compile_per_spec_across_masks(synthetic_ssp):
    """A second batch of the same spec, with different line masks, triggers no compile."""
    from jax._src import test_util as jtu

    from tengri.inference.batched_spectro import batched_log_likelihood
    from tengri.observation.batched import build_spectro_batches

    model = _marg_template(synthetic_ssp)
    spec_a, batch_a, _ = build_spectro_batches(
        [_marg_galaxy(0.1), _marg_galaxy(0.12, seed=1)], model=model
    )[0]
    spec_b, batch_b, _ = build_spectro_batches(
        [_marg_galaxy(0.2), _marg_galaxy(0.3, seed=1)], model=model
    )[0]
    assert spec_a == spec_b
    assert not np.array_equal(np.asarray(batch_a.line_mask), np.asarray(batch_b.line_mask))

    loglik = jax.jit(jax.vmap(batched_log_likelihood(model, spec_a), in_axes=(0, 0)))
    params_a = {"sfh_dpl_log_total_mass": jnp.full((2,), 10.0)}
    params_b = {"sfh_dpl_log_total_mass": jnp.full((2,), 10.0)}
    jax.block_until_ready(loglik(params_a, batch_a))
    with jtu.count_jit_compilation_cache_miss() as counter:
        jax.block_until_ready(loglik(params_b, batch_b))
    assert counter() == 0


def test_padding_quantum_leaves_marginal_unchanged(synthetic_ssp):
    """quantum=None and quantum=64 give the same marginal at rtol 1e-10 (banded LSF)."""
    from tengri.inference.batched_spectro import batched_log_likelihood
    from tengri.observation.batched import build_spectro_batches

    model = _marg_template(synthetic_ssp, banded=True)
    banded = model.observation.spectroscopy.resolution_matrix
    params = {"sfh_dpl_log_total_mass": 10.0}
    values = []
    for quantum in (None, 64):
        spec, batch, _ = build_spectro_batches(
            [_marg_galaxy(0.1, resolution=banded)], model=model, quantum=quantum
        )[0]
        values.append(float(batched_log_likelihood(model, spec)(params, batch.galaxy(0))))
    np.testing.assert_allclose(values[1], values[0], rtol=1e-10)


def _model_matched_data(ssp, z):
    """Data in the model's flux units for a 10^10 Msun galaxy: (flux, sigma, prior_sigma).

    The truth is the float64 spectrum of the same template at log10 M = 10. The data is
    the truth plus noise at the level of sigma, and sigma is 10% of the truth plus a floor of
    1e-2 of its peak, so sigma stays representable in float32. The line prior width is the peak,
    the same ratio to the continuum as the 1e-15 width had to the 1.3e-15 flux.
    """
    truth = np.asarray(
        _marg_template(ssp).predict_spectrum({"sfh_dpl_log_total_mass": 10.0, "redshift": z})
    )
    peak = float(np.max(np.abs(truth)))
    sigma = 0.1 * np.abs(truth) + 1.0e-2 * peak
    noise = np.random.default_rng(0).standard_normal(truth.size)
    return truth + sigma * noise, sigma, peak


_F32_LOG_M = 10.1  # offset from the truth, so the residuals are nonzero


def _batched_value_and_grad(ssp, flux, sigma, z, prior_sigma, log_mass):
    """Batched marginal value and gradient dict at one point, in the current precision."""
    from tengri.inference.batched_spectro import batched_log_likelihood
    from tengri.observation.batched import GalaxySpectrum, build_spectro_batches

    model = _marg_template(ssp, prior_sigma=prior_sigma)
    gal = GalaxySpectrum(wave=_MARG_WAVE, flux=flux, ivar=1.0 / sigma**2, z=z, resolution=_MARG_R)
    spec, batch, _ = build_spectro_batches([gal], model=model)[0]
    loglik = batched_log_likelihood(model, spec)
    obs = batch.galaxy(0)
    params = {"sfh_dpl_log_total_mass": log_mass}
    value = loglik(params, obs)
    grad = jax.grad(lambda p: loglik(p, obs))(params)
    return float(value), grad


def test_float32_marginal_is_finite_and_nonzero(synthetic_ssp_physical):
    """With x64 off, the batched marginal and its gradient are finite and nonzero.

    The galaxy is 10^10 Msun on the physically normalized SSP, with data in the model's flux
    units. The float32 value and gradient must also agree with float64 at the same point.
    """
    z = 0.1
    flux, sigma, prior_sigma = _model_matched_data(synthetic_ssp_physical, z)
    value64, grad64 = _batched_value_and_grad(
        synthetic_ssp_physical, flux, sigma, z, prior_sigma, _F32_LOG_M
    )
    with jax.enable_x64(False):
        value32, grad32 = _batched_value_and_grad(
            synthetic_ssp_physical, flux, sigma, z, prior_sigma, _F32_LOG_M
        )
    assert np.isfinite(value32)
    assert value32 != 0.0
    for leaf in jax.tree.leaves(grad32):
        assert np.all(np.isfinite(np.asarray(leaf)))
        assert np.all(np.asarray(leaf) != 0.0)
    rel_value = abs(value32 - value64) / abs(value64)
    rel_grad = max(
        abs(float(np.asarray(g32)) - float(np.asarray(g64))) / abs(float(np.asarray(g64)))
        for g32, g64 in zip(jax.tree.leaves(grad32), jax.tree.leaves(grad64))
    )
    # Measured float32 vs float64 at this point: rel_value 3.04e-5, rel_grad 2.41e-5.
    assert rel_value < 1.5e-4
    assert rel_grad < 1.5e-4


def test_fitted_lines_with_calibration_marginal_are_refused(synthetic_ssp):
    """Mirrors the per-galaxy refusal: fitted lines cannot be combined with calibration."""
    from tengri.inference.batched_spectro import batched_log_likelihood
    from tengri.observation.batched import build_spectro_batches

    model = _marg_template(synthetic_ssp, eline_mode="fitted")
    spec, _, _ = build_spectro_batches([_marg_galaxy(0.1)], model=model)[0]
    with pytest.raises(NotImplementedError, match="fitted"):
        batched_log_likelihood(model, spec, calibration_marginalize=True)


# Per-galaxy marginalized classes in float32 at the batched test's point (#3117, #3149).
#
# The stellar mass is log10 M = 10 on the physically normalized SSP, and the flux, noise
# and prior width are the double-precision values. The old formula formed 1/sigma_p**2,
# which is non-finite in float32 at this scale.


def _per_galaxy_classes(template, z, *, flux, sigma, prior_sigma):
    """Lines-only and joint per-galaxy likelihoods at the given data, prior and redshift."""
    from tengri.inference.likelihoods.marginalized import (
        CalibrationELineMarginalizedLikelihood,
        ELineMarginalizedLikelihood,
    )

    lines = ELineMarginalizedLikelihood(
        fnu_obs=jnp.asarray(flux),
        fnu_err=jnp.asarray(sigma),
        design_matrix=_marg_line_design(template, z),
        prior_sigma=prior_sigma,
    )
    joint = CalibrationELineMarginalizedLikelihood(
        fnu_obs=jnp.asarray(flux),
        fnu_err=jnp.asarray(sigma),
        wavelength=jnp.asarray(_MARG_WAVE),
        design_matrix_builder=lambda _p: _marg_line_design(template, z),
        n_poly=2,
        prior_sigma=1.0,
        eline_prior_type="flat",
        eline_prior_sigma=prior_sigma,
        channel="spec_fnu",
    )
    return lines, joint


def test_per_galaxy_marginal_classes_are_finite_and_match_float64_in_float32(
    synthetic_ssp_physical,
):
    """Lines-only and joint per-galaxy marginals are finite in float32 and match float64.

    At log10 M = 10 on the physically normalized SSP, the old 1/sigma_p**2 form is
    non-finite in float32. The float32 values must agree with the float64 values to
    rtol 2e-5. The prediction is the float64 model cast to float32, so the comparison
    isolates the marginal; the float32 forward is covered by the batched float32 test.
    """
    z = 0.1
    params = {"sfh_dpl_log_total_mass": 10.0}
    flux, sigma = _marg_data()
    prior_sigma = _MARG_PRIOR_SIGMA
    template64 = _marg_template(synthetic_ssp_physical)
    m64 = jnp.asarray(template64.predict_spectrum({**params, "redshift": z}))
    lines64, joint64 = _per_galaxy_classes(
        template64,
        z,
        flux=flux,
        sigma=sigma,
        prior_sigma=prior_sigma,
    )
    expected_lines = float(lines64.log_prob({"spec_fnu": m64}))
    expected_joint = float(joint64.log_prob({"spec_fnu": m64}, params={}))

    with jax.enable_x64(False):
        template32 = _marg_template(synthetic_ssp_physical, prior_sigma=prior_sigma)
        m32 = jnp.asarray(m64)
        lines32, joint32 = _per_galaxy_classes(
            template32,
            z,
            flux=flux,
            sigma=sigma,
            prior_sigma=prior_sigma,
        )
        got_lines = float(lines32.log_prob({"spec_fnu": m32}))
        got_joint = float(joint32.log_prob({"spec_fnu": m32}, params={}))

    assert np.isfinite(got_lines)
    assert np.isfinite(got_joint)
    np.testing.assert_allclose(got_lines, expected_lines, rtol=2e-5)
    np.testing.assert_allclose(got_joint, expected_joint, rtol=2e-5)


def _per_galaxy_value_and_grad(ssp, z, flux, sigma, prior_sigma, log_mass, *, joint):
    """Per-galaxy marginal value and d/d log10 M at one point, in the current precision."""
    template = _marg_template(ssp, prior_sigma=prior_sigma)
    lines, joint_cls = _per_galaxy_classes(
        template, z, flux=flux, sigma=sigma, prior_sigma=prior_sigma
    )
    cls = joint_cls if joint else lines

    def value(logm):
        m = jnp.asarray(template.predict_spectrum({"sfh_dpl_log_total_mass": logm, "redshift": z}))
        if joint:
            return cls.log_prob({"spec_fnu": m}, params={})
        return cls.log_prob({"spec_fnu": m})

    return float(value(jnp.asarray(_F32_LOG_M))), float(jax.grad(value)(jnp.asarray(_F32_LOG_M)))


def test_per_galaxy_marginal_gradients_are_finite_and_nonzero_in_float32(synthetic_ssp_physical):
    """Lines-only and joint per-galaxy marginals have finite, nonzero gradients in float32.

    The galaxy is 10^10 Msun on the physically normalized SSP, with data in the model's flux
    units. The float32 values and gradients must also agree with float64 at the same point.
    """
    z = 0.1
    flux, sigma, prior_sigma = _model_matched_data(synthetic_ssp_physical, z)
    for joint in (False, True):
        value64, grad64 = _per_galaxy_value_and_grad(
            synthetic_ssp_physical, z, flux, sigma, prior_sigma, _F32_LOG_M, joint=joint
        )
        with jax.enable_x64(False):
            value32, grad32 = _per_galaxy_value_and_grad(
                synthetic_ssp_physical, z, flux, sigma, prior_sigma, _F32_LOG_M, joint=joint
            )
        assert np.isfinite(value32)
        assert value32 != 0.0
        assert np.isfinite(grad32)
        assert grad32 != 0.0
        rel_value = abs(value32 - value64) / abs(value64)
        rel_grad = abs(grad32 - grad64) / abs(grad64)
        # Measured float32 vs float64 at this point, lines-only and joint: rel_value up to
        # 5.79e-5, rel_grad up to 6.87e-5.
        assert rel_value < 4.0e-4
        assert rel_grad < 4.0e-4


def test_per_galaxy_and_batched_marginals_both_call_the_one_helper(synthetic_ssp, monkeypatch):
    """Replacing the shared line-prior helper makes both paths raise, so there is one helper."""
    import tengri.inference.batched_spectro as batched_spectro
    import tengri.inference.likelihoods.marginalized as marginalized
    import tengri.observation.eline_marginalization as eline_marginalization
    from tengri.inference.batched_spectro import batched_log_likelihood
    from tengri.observation.batched import GalaxySpectrum, build_spectro_batches

    def _refuse(*_args, **_kwargs):
        raise RuntimeError("shared line-prior helper called")

    z = 0.1
    template = _marg_template(synthetic_ssp)
    flux, sigma = _marg_data()
    gal = GalaxySpectrum(wave=_MARG_WAVE, flux=flux, ivar=1.0 / sigma**2, z=z, resolution=_MARG_R)
    spec, batch, _ = build_spectro_batches([gal], model=template)[0]
    params = {"sfh_dpl_log_total_mass": 10.0}
    m_single = template.predict_spectrum({**params, "redshift": z})
    lines, joint = _per_galaxy_classes(
        template, z, flux=flux, sigma=sigma, prior_sigma=_MARG_PRIOR_SIGMA
    )
    loglik = batched_log_likelihood(template, spec)

    monkeypatch.setattr(eline_marginalization, "unit_line_prior", _refuse)
    monkeypatch.setattr(marginalized, "unit_line_prior", _refuse)
    monkeypatch.setattr(batched_spectro, "unit_line_prior", _refuse)
    jax.clear_caches()
    with pytest.raises(RuntimeError, match="shared line-prior helper"):
        lines.log_prob({"spec_fnu": m_single})
    with pytest.raises(RuntimeError, match="shared line-prior helper"):
        joint.log_prob({"spec_fnu": m_single}, params={})
    with pytest.raises(RuntimeError, match="shared line-prior helper"):
        loglik(params, batch.galaxy(0))
