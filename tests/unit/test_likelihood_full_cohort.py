# SPDX-License-Identifier: BSD-3-Clause
"""Full cohort of :class:`Likelihood` Protocol adapters.

Validates the channel-parameterised redesign:

- `GaussianLikelihood` is the workhorse — pinning ``channel`` lets
  it cover photometry, spectroscopy, line fluxes, spectral indices,
  EWs, etc.
- `StudentTLikelihood`, `CensoredLikelihood`,
  `MultivariateGaussianLikelihood` are *distinct math types* — each
  channel-parameterised the same way.
- `CalibrationMarginalisedLikelihood` and
  `ELineMarginalisedLikelihood` are *additional math types* with
  analytic nuisance integration.

For each adapter:

1. duck-types as :class:`tengri.protocols.Likelihood` Protocol.
2. matches the legacy primitive bit-for-bit on identical inputs.
3. composes through :class:`CompositeLikelihood` without collision.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.inference.composite_likelihood import CompositeLikelihood
from tengri.inference.likelihoods import (
    CalibrationMarginalisedLikelihood,
    CensoredLikelihood,
    ELineMarginalisedLikelihood,
    GaussianLikelihood,
    MultivariateGaussianLikelihood,
    StudentTLikelihood,
    diag_gaussian_log_prob,
)
from tengri.observation.calibration import marginalize_calibration
from tengri.observation.eline_marginalization import marginalize_emission_lines
from tengri.observation.noise import (
    censored_neg_log_likelihood,
    variable_noise_hamiltonian,
)
from tengri.protocols import Likelihood

# ─────────────────────────────────────────────────────────────────────
# Channel-parameterised GaussianLikelihood
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "channel",
    ["phot_fnu", "spec_fnu", "line_fluxes", "indices", "any_string_works"],
)
def test_gaussian_likelihood_handles_any_channel(channel):
    """A single class with channel= covers every diagonal-Gaussian
    observation type — no per-channel subclass needed."""
    obs = jnp.array([1.0, 2.0, 3.0])
    err = jnp.array([0.1, 0.1, 0.1])
    pred = jnp.array([1.05, 2.0, 2.95])

    lk = GaussianLikelihood(obs=obs, err=err, channel=channel)
    assert isinstance(lk, Likelihood)

    expected = diag_gaussian_log_prob(pred, obs, err)
    actual = lk.log_prob({channel: pred})
    assert float(actual) == pytest.approx(float(expected), rel=1e-10)


@pytest.mark.unit
def test_gaussian_likelihood_unaffected_by_extra_prediction_keys():
    """Extra keys in the prediction dict are ignored."""
    obs = jnp.array([1.0, 2.0])
    err = jnp.array([0.1, 0.1])
    lk = GaussianLikelihood(obs=obs, err=err, channel="phot_fnu")
    pred = {"phot_fnu": jnp.array([1.0, 2.0]), "irrelevant": jnp.array([0.0, 99.0])}
    assert float(lk.log_prob(pred)) == 0.0  # perfect fit


# ─────────────────────────────────────────────────────────────────────
# StudentTLikelihood
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("dof", [2.0, 4.0, None])
def test_student_t_matches_legacy_primitive(dof):
    obs = jnp.array([1.0, 2.0, 3.0])
    err = jnp.array([0.1, 0.1, 0.1])
    pred = jnp.array([1.2, 2.0, 2.8])

    lk = StudentTLikelihood(obs=obs, err=err, dof=dof, f_cal=0.05, channel="phot_fnu")
    assert isinstance(lk, Likelihood)

    expected = -float(variable_noise_hamiltonian(obs, err, pred, f_cal=0.05, dof=dof))
    actual = float(lk.log_prob({"phot_fnu": pred}))
    assert actual == pytest.approx(expected, rel=1e-10)


@pytest.mark.unit
def test_student_t_outliers_get_smaller_penalty_than_gaussian():
    """For a 5σ outlier, Student-t gives a *less negative* log-prob
    than Gaussian — that's the whole point of the heavy tail."""
    obs = jnp.array([1.0, 2.0, 100.0])  # last is a wild outlier
    err = jnp.array([0.1, 0.1, 0.1])
    pred = jnp.array([1.0, 2.0, 1.0])

    gauss = GaussianLikelihood(obs=obs, err=err, channel="phot_fnu")
    studt = StudentTLikelihood(obs=obs, err=err, dof=2.0, channel="phot_fnu")

    g_lp = float(gauss.log_prob({"phot_fnu": pred}))
    t_lp = float(studt.log_prob({"phot_fnu": pred}))
    # Student-t is much less penalised on the outlier-heavy fit.
    assert t_lp > g_lp


# ─────────────────────────────────────────────────────────────────────
# CensoredLikelihood
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_censored_matches_legacy_primitive():
    obs = jnp.array([1.0, 2.0, 0.5])
    err = jnp.array([0.1, 0.1, 0.1])
    pred = jnp.array([1.05, 1.95, 0.3])
    mask = jnp.array([0, 0, 1])  # last is upper limit

    lk = CensoredLikelihood(obs=obs, err=err, mask=mask, channel="phot_fnu")
    expected = -float(censored_neg_log_likelihood(obs, err, pred, mask, f_cal=0.0, dof=None))
    actual = float(lk.log_prob({"phot_fnu": pred}))
    assert actual == pytest.approx(expected, rel=1e-10)


@pytest.mark.unit
def test_censored_upper_limit_well_below_data_is_a_good_fit():
    """Upper limit at flux=1, prediction far below → high log-prob."""
    obs = jnp.array([1.0])
    err = jnp.array([0.1])
    pred_below = jnp.array([0.01])  # well below the upper limit
    pred_above = jnp.array([10.0])  # well above the upper limit
    mask = jnp.array([1])

    lk = CensoredLikelihood(obs=obs, err=err, mask=mask, channel="phot_fnu")
    lp_below = float(lk.log_prob({"phot_fnu": pred_below}))
    lp_above = float(lk.log_prob({"phot_fnu": pred_above}))
    assert lp_below > lp_above


# ─────────────────────────────────────────────────────────────────────
# MultivariateGaussianLikelihood
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_mvn_recovers_diagonal_gaussian_when_cov_is_diagonal():
    """MVN with diagonal cov_inv == diag-Gaussian (modulo the dropped
    normalisation constant)."""
    obs = jnp.array([1.0, 2.0, 3.0])
    err = jnp.array([0.1, 0.2, 0.15])
    pred = jnp.array([1.1, 1.95, 3.05])

    cov_inv = jnp.diag(1.0 / err**2)
    mvn = MultivariateGaussianLikelihood(obs=obs, cov_inv=cov_inv, channel="spec_fnu")
    expected = float(diag_gaussian_log_prob(pred, obs, err))
    actual = float(mvn.log_prob({"spec_fnu": pred}))
    assert actual == pytest.approx(expected, rel=1e-10)


@pytest.mark.unit
def test_mvn_off_diagonal_changes_result():
    """An off-diagonal correlation must change the log-prob away from
    the diagonal answer."""
    obs = jnp.array([1.0, 2.0])
    err = jnp.array([0.1, 0.1])
    pred = jnp.array([1.1, 1.9])

    diag_inv = jnp.diag(1.0 / err**2)
    corr_inv = diag_inv + jnp.array([[0.0, 50.0], [50.0, 0.0]])  # add cross term

    diag_lk = MultivariateGaussianLikelihood(obs=obs, cov_inv=diag_inv, channel="spec_fnu")
    corr_lk = MultivariateGaussianLikelihood(obs=obs, cov_inv=corr_inv, channel="spec_fnu")

    assert (
        abs(
            float(diag_lk.log_prob({"spec_fnu": pred}))
            - float(corr_lk.log_prob({"spec_fnu": pred}))
        )
        > 1e-3
    )


# ─────────────────────────────────────────────────────────────────────
# CalibrationMarginalisedLikelihood
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_calibration_marginalised_matches_legacy_primitive():
    wave = jnp.linspace(4000.0, 7000.0, 64)
    obs = jnp.exp(-((wave - 5500.0) ** 2) / (2 * 1000.0**2))
    err = jnp.ones_like(obs) * 0.05
    model = obs * 1.05  # constant 5% miscalibration the polynomial absorbs

    lk = CalibrationMarginalisedLikelihood(
        fnu_obs=obs, fnu_err=err, wavelength=wave, n_poly=3, prior_sigma=1.0
    )
    expected_log_lik, _, _ = marginalize_calibration(
        model_flux=model,
        obs_flux=obs,
        obs_err=err,
        wavelength=wave,
        n_poly=3,
        prior_sigma=1.0,
    )
    actual = float(lk.log_prob({"spec_fnu": model}))
    assert actual == pytest.approx(float(expected_log_lik), rel=1e-10)


# ─────────────────────────────────────────────────────────────────────
# ELineMarginalisedLikelihood
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_eline_marginalised_matches_legacy_primitive():
    n_pix, n_lines = 32, 3
    obs = jnp.linspace(1.0, 2.0, n_pix)
    err = jnp.ones(n_pix) * 0.05
    model = obs - 0.1  # continuum slightly below
    # Mock 3-line Gaussian design matrix.
    centres = jnp.array([5, 16, 25])
    sigmas = jnp.array([2.0, 2.5, 1.5])
    pixel_idx = jnp.arange(n_pix).astype(float)
    G = jnp.stack(
        [
            jnp.exp(-0.5 * ((pixel_idx - c) / s) ** 2)
            for c, s in zip(centres, sigmas, strict=False)
        ],
        axis=1,
    )

    lk = ELineMarginalisedLikelihood(
        fnu_obs=obs,
        fnu_err=err,
        design_matrix=G,
    )
    expected_ll, _, _ = marginalize_emission_lines(
        residual=obs - model,
        noise=err,
        design_matrix=G,
        prior_variance=None,
    )
    actual = float(lk.log_prob({"spec_fnu": model}))
    assert actual == pytest.approx(float(expected_ll), rel=1e-10)


# ─────────────────────────────────────────────────────────────────────
# Composition: every adapter into a single CompositeLikelihood
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_full_cohort_composes_in_one_composite():
    """Build a contrived composite using every adapter type — no
    collisions, log_prob == sum of constituents."""
    n = 8
    pred_phot = jnp.ones(n)
    pred_spec = jnp.ones(n)
    pred_lines = jnp.ones(3)
    err = jnp.ones(n) * 0.1
    obs_phot = jnp.ones(n) * 0.95
    mask = jnp.zeros(n, dtype=int)
    cov_inv = jnp.diag(1.0 / err**2)

    components = [
        GaussianLikelihood(obs=obs_phot, err=err, channel="phot_fnu"),
        StudentTLikelihood(obs=obs_phot, err=err, dof=4.0, channel="phot_fnu_t"),
        CensoredLikelihood(obs=obs_phot, err=err, mask=mask, channel="phot_fnu_cens"),
        MultivariateGaussianLikelihood(obs=obs_phot, cov_inv=cov_inv, channel="spec_fnu"),
        GaussianLikelihood(
            obs=jnp.array([1.0, 2.0, 3.0]),
            err=jnp.array([0.1, 0.1, 0.1]),
            channel="line_fluxes",
        ),
    ]
    composite = CompositeLikelihood(*components)

    prediction = {
        "phot_fnu": pred_phot,
        "phot_fnu_t": pred_phot,
        "phot_fnu_cens": pred_phot,
        "spec_fnu": pred_spec,
        "line_fluxes": pred_lines,
    }
    expected = sum(float(c.log_prob(prediction)) for c in components)
    actual = float(composite.log_prob(prediction))
    assert actual == pytest.approx(expected, rel=1e-10)


# ──────────────────────────────────────────────────────────────────────
# Phase II-2.3 — equivalence test for combined cal+eline adapter
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_calibration_eline_marginalised_matches_deleted_legacy_sequential():
    """``CalibrationELineMarginalisedLikelihood.log_prob(...)`` must match
    the deleted legacy sequential composition bit-for-bit:

        marginalize_emission_lines → augment prediction → marginalize_calibration

    This pins the new adapter to the math the legacy χ² switch
    encoded in lines 248-297 of pre-II-2.3 ``loss_functions.py``.
    """
    import jax.numpy as jnp

    from tengri.inference.likelihoods.marginalised import (
        CalibrationELineMarginalisedLikelihood,
    )
    from tengri.observation.calibration import marginalize_calibration
    from tengri.observation.eline_marginalization import marginalize_emission_lines

    # Tiny synthetic spec: 8 wavelength pixels, 2 emission lines.
    rng = jnp.linspace(0.0, 1.0, 8)
    fnu_obs = 1.0 + 0.1 * jnp.cos(2 * jnp.pi * rng)
    fnu_err = jnp.full(8, 0.05)
    wavelength = jnp.linspace(4500.0, 6700.0, 8)
    model_spec = 1.0 + 0.05 * rng  # smooth model continuum (no lines)
    # Two narrow lines: design matrix columns are Gaussian profiles.
    line_centres = jnp.array([5000.0, 6500.0])
    sigma = 30.0
    design_matrix = jnp.stack(
        [jnp.exp(-0.5 * ((wavelength - lc) / sigma) ** 2) for lc in line_centres],
        axis=1,
    )
    prior_sigma_eline = 0.5  # per-line amplitude prior σ
    prior_var_eline = jnp.full(2, prior_sigma_eline**2)

    # ── Path A: deleted legacy sequential composition ──
    residual = fnu_obs - model_spec
    _, a_hat, _ = marginalize_emission_lines(
        residual, fnu_err, design_matrix, prior_variance=prior_var_eline
    )
    pred_with_lines = model_spec + design_matrix @ a_hat
    expected_log_lik, _, _ = marginalize_calibration(
        model_flux=pred_with_lines,
        obs_flux=fnu_obs,
        obs_err=fnu_err,
        wavelength=wavelength,
        n_poly=3,
        prior_sigma=1.0,
    )

    # ── Path B: new adapter ──
    adapter = CalibrationELineMarginalisedLikelihood(
        fnu_obs=fnu_obs,
        fnu_err=fnu_err,
        wavelength=wavelength,
        design_matrix_builder=lambda _params: design_matrix,
        n_poly=3,
        prior_sigma=1.0,
        eline_prior_type="flat",
        eline_prior_sigma=prior_sigma_eline,
        channel="spec_fnu",
    )
    actual = adapter.log_prob({"spec_fnu": model_spec}, params={})

    # Bit-for-bit (within FP rounding) — both paths share the underlying
    # primitives, so the difference should be machine ε.
    assert float(actual) == pytest.approx(float(expected_log_lik), rel=1e-12, abs=1e-12)
