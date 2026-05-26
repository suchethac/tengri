from __future__ import annotations

import chex
import pytest

# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Fitter auto-build path (option β default).

Validates that ``Fitter(model, data, noise)`` automatically constructs
the matching :class:`Likelihood` Protocol object when no legacy-only
features (calibration marginalisation, e-line marginalisation,
Student-t / variable noise, spec covariance, censored data, line
fluxes, spectral indices) are configured.

Three data_type cases:

- photometry → :class:`PhotometryLikelihood`
- spectroscopy → :class:`SpectroscopyLikelihood`
- joint → :class:`CompositeLikelihood` of phot + spec, split at
  ``observation.n_data_phot``.

The numerical equivalence with the legacy χ² dispatch is enforced by
the :func:`build_loss_fn` regression suite, which still passes after
auto-build is enabled (``243 passed`` 2026-05-03).
"""


from types import SimpleNamespace

import jax.numpy as jnp

from tengri.inference.composite_likelihood import CompositeLikelihood
from tengri.inference.fitter import Fitter
from tengri.inference.photometry_likelihood import PhotometryLikelihood
from tengri.inference.spectroscopy_likelihood import SpectroscopyLikelihood

pytestmark = pytest.mark.contract

# ──────────────────────────────────────────────────────────────────────
# Mock Fitter — bypasses the real __init__ to focus on the helper.
# ──────────────────────────────────────────────────────────────────────


def _make_helper_only_fitter(
    *,
    data,
    noise,
    data_type,
    has_eline=False,
    has_cal=False,
    has_mask=False,
    n_data_phot=None,
):
    """Build a thin proxy carrying every attribute the auto-build path
    reads.

    Likelihood builders (``build_base_likelihood``, ``build_likelihood_extras``)
    are invoked via the likelihood module so we don't need a real :class:`Fitter`
    instance (which spawns a background compile thread we don't want for unit tests).
    """
    spec = SimpleNamespace(all_params=[], free_params=[])
    obs = SimpleNamespace(n_data_phot=n_data_phot)
    model = SimpleNamespace(observation=obs)
    return SimpleNamespace(
        data=jnp.asarray(data),
        noise=jnp.asarray(noise),
        data_type=data_type,
        data_mask=jnp.asarray([1, 1, 1]) if has_mask else None,
        spec=spec,
        model=model,
        _calibration_marginalize=has_cal,
        _has_spectroscopy=data_type in ("spectroscopy", "joint"),
        _eline_marginalize=has_eline,
        _eline_fitted=False,
        _eline_prior_type=None,
        _eline_wavelengths=None,
        _eline_constraint_matrix=None,
        _cal_n_poly=3,
        _cal_prior_sigma=1.0,
        _fixed_values={},
        _data_args={"data": jnp.asarray(data), "noise": jnp.asarray(noise)},
    )


def _make_proxy(**kw):
    """Make the proxy and self-reference it so bound lambdas work.

    Wraps the proxy in an :class:`InferenceContext` before handing it to
    the likelihood builders — the Step-D-prime refactor (ADR-0009)
    changed the builder signatures from ``(fitter)`` to
    ``(context: InferenceContext)``. The context's properties delegate
    to the proxy's underscore-prefixed attributes transparently.
    """
    from tengri.inference.context import InferenceContext
    from tengri.inference.likelihood import build_base_likelihood, build_likelihood_extras

    proxy = _make_helper_only_fitter(**kw)
    context = InferenceContext.from_target(proxy)
    # Re-bind helpers to the actual proxy via the extracted likelihood module
    proxy._build_base_likelihood = lambda: build_base_likelihood(context)
    proxy._build_likelihood_extras = lambda: build_likelihood_extras(context)
    return proxy


# ──────────────────────────────────────────────────────────────────────
# Auto-build cases
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_photometry_auto_builds_phot_likelihood():
    fitter = _make_proxy(
        data=[1e-29, 2e-29, 3e-29],
        noise=[0.1e-29, 0.1e-29, 0.1e-29],
        data_type="photometry",
    )
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, PhotometryLikelihood)
    assert jnp.allclose(lk.fnu_obs, jnp.asarray([1e-29, 2e-29, 3e-29]))


@pytest.mark.unit
def test_spectroscopy_auto_builds_spec_likelihood():
    fitter = _make_proxy(
        data=[1.0, 1.5, 2.0],
        noise=[0.05, 0.05, 0.05],
        data_type="spectroscopy",
    )
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, SpectroscopyLikelihood)
    assert jnp.allclose(lk.fnu_obs, jnp.asarray([1.0, 1.5, 2.0]))


@pytest.mark.unit
def test_joint_auto_builds_composite_with_data_split():
    """Joint data is split at observation.n_data_phot into phot+spec
    likelihoods wrapped in a CompositeLikelihood."""
    n_phot = 3
    fitter = _make_proxy(
        data=[1e-29, 2e-29, 3e-29, 1.0, 1.5, 2.0, 2.5, 3.0],
        noise=[0.1e-29, 0.1e-29, 0.1e-29, 0.05, 0.05, 0.05, 0.05, 0.05],
        data_type="joint",
        n_data_phot=n_phot,
    )
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, CompositeLikelihood)
    assert len(lk.likelihoods) == 2
    phot, spec = lk.likelihoods
    assert isinstance(phot, PhotometryLikelihood)
    assert isinstance(spec, SpectroscopyLikelihood)
    chex.assert_shape(phot.fnu_obs, (n_phot,))
    chex.assert_shape(spec.fnu_obs, (5,))


# ──────────────────────────────────────────────────────────────────────
# Each previously-falls-back case is now a built adapter
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_calibration_marginalize_auto_builds_calibration_likelihood():
    """Phase II-1 final wiring: cal-marg → CalibrationMarginalisedLikelihood."""
    from tengri.inference.likelihoods import CalibrationMarginalisedLikelihood

    fitter = _make_proxy(
        data=[1.0, 2.0],
        noise=[0.1, 0.1],
        data_type="spectroscopy",
        has_cal=True,
    )
    fitter.model.wave_obs = jnp.array([5000.0, 6000.0])
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, CalibrationMarginalisedLikelihood)
    assert lk.n_poly == 3
    assert lk.channel == "spec_fnu"


@pytest.mark.unit
def test_eline_marginalize_auto_builds_eline_likelihood_with_builder():
    """Flat-prior e-line marg → ELineMarginalisedLikelihood with a
    per-call design_matrix_builder closure."""
    from tengri.inference.likelihoods import ELineMarginalisedLikelihood

    fitter = _make_proxy(
        data=[1.0, 2.0, 3.0],
        noise=[0.1, 0.1, 0.1],
        data_type="spectroscopy",
        has_eline=True,
    )
    fitter._eline_wavelengths = jnp.array([6564.6, 4861.3])
    fitter._eline_constraint_matrix = jnp.eye(2)
    fitter.model.wave_obs = jnp.linspace(4000.0, 8000.0, 3)
    fitter.model._spectral_resolution = 2000.0
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, ELineMarginalisedLikelihood)
    assert lk.design_matrix_builder is not None


@pytest.mark.unit
def test_eline_cloudy_prior_auto_builds_cloudy_likelihood():
    """Cloudy-prior e-line marg → CloudyELineMarginalisedLikelihood
    (was previously a legacy fall-back; now covered by the adapter
    cohort post Phase II-2.2 migration)."""
    from tengri.inference.likelihoods import CloudyELineMarginalisedLikelihood

    fitter = _make_proxy(
        data=[1.0, 2.0, 3.0],
        noise=[0.1, 0.1, 0.1],
        data_type="spectroscopy",
        has_eline=True,
    )
    fitter._eline_prior_type = "cloudy"
    fitter._eline_wavelengths = jnp.array([6564.6, 4861.3])
    fitter._eline_independent_wavelengths = jnp.array([6564.6, 4861.3])
    fitter._eline_constraint_matrix = jnp.eye(2)
    fitter._eline_prior_width_dex = 0.5
    fitter.model.wave_obs = jnp.linspace(4000.0, 8000.0, 3)
    fitter.model._spectral_resolution = 2000.0
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, CloudyELineMarginalisedLikelihood)
    assert lk.channel == "spec_fnu"


@pytest.mark.unit
def test_censored_data_auto_builds_censored_likelihood():
    """data_mask present → CensoredLikelihood."""
    from tengri.inference.likelihoods import CensoredLikelihood

    fitter = _make_proxy(
        data=[1.0, 2.0, 3.0],
        noise=[0.1, 0.1, 0.1],
        data_type="photometry",
        has_mask=True,
    )
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, CensoredLikelihood)
    assert lk.channel == "phot_fnu"


@pytest.mark.unit
@pytest.mark.parametrize("data_type", ["spectroscopy", "joint"])
def test_censored_data_on_non_photometry_falls_back_to_legacy(data_type):
    """data_mask + spec/joint → defer to legacy χ² path.

    Regression: the auto-build only builds CensoredLikelihood for
    photometry (channel="phot_fnu"). For spec/joint, the mask spans
    the concatenated data array and is not addressable via a
    single-channel adapter — bailing to None lets the legacy
    `use_censored` branch apply censored_neg_log_likelihood across the
    whole prediction. Without this bail-out, downstream branches
    silently built a plain SpectroscopyLikelihood / Composite that
    ignored the mask.
    """
    fitter = _make_proxy(
        data=[1.0, 2.0, 3.0],
        noise=[0.1, 0.1, 0.1],
        data_type=data_type,
        has_mask=True,
    )
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert lk is None


@pytest.mark.unit
def test_spec_cov_auto_builds_multivariate_gaussian():
    """spec_cov_inv data → MultivariateGaussianLikelihood."""
    from tengri.inference.likelihoods import MultivariateGaussianLikelihood

    fitter = _make_proxy(
        data=[1.0, 2.0],
        noise=[0.1, 0.1],
        data_type="spectroscopy",
    )
    fitter._data_args["spec_cov_inv"] = jnp.eye(2)
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, MultivariateGaussianLikelihood)
    assert lk.channel == "spec_fnu"


@pytest.mark.unit
def test_line_fluxes_compose_into_composite_with_base():
    """line_flux_waves present → base + GaussianLikelihood(channel='line_fluxes')."""
    from tengri.inference.likelihoods import GaussianLikelihood

    fitter = _make_proxy(
        data=[1.0, 2.0],
        noise=[0.1, 0.1],
        data_type="spectroscopy",
    )
    fitter._data_args["line_flux_waves"] = jnp.array([6564.6])
    fitter._data_args["line_flux_obs"] = jnp.array([1e-15])
    fitter._data_args["line_flux_err"] = jnp.array([1e-16])
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, CompositeLikelihood)
    assert len(lk.likelihoods) == 2
    assert isinstance(lk.likelihoods[0], SpectroscopyLikelihood)
    extra = lk.likelihoods[1]
    assert isinstance(extra, GaussianLikelihood)
    assert extra.channel == "line_fluxes"


@pytest.mark.unit
def test_spectral_indices_compose_into_composite_with_base():
    """index_obs present → base + GaussianLikelihood(channel='indices')."""
    from tengri.inference.likelihoods import GaussianLikelihood

    fitter = _make_proxy(
        data=[1.0, 2.0],
        noise=[0.1, 0.1],
        data_type="spectroscopy",
    )
    fitter._data_args["index_obs"] = jnp.array([1.5])
    fitter._data_args["index_err"] = jnp.array([0.05])
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, CompositeLikelihood)
    indices_lk = next(
        c for c in lk.likelihoods if isinstance(c, GaussianLikelihood) and c.channel == "indices"
    )
    chex.assert_shape(indices_lk.obs, (1,))


@pytest.mark.unit
def test_eline_fitted_auto_builds_fitted_likelihood():
    """Explicitly-fitted e-line amplitudes → ELineFittedLikelihood
    (was previously a legacy fall-back; now covered by the adapter
    cohort post Phase II-2.2 migration)."""
    from tengri.inference.likelihoods import ELineFittedLikelihood

    fitter = _make_proxy(
        data=[1.0, 2.0, 3.0],
        noise=[0.1, 0.1, 0.1],
        data_type="spectroscopy",
    )
    fitter._eline_fitted = True
    fitter._eline_wavelengths = jnp.array([6564.6, 4861.3])
    fitter._eline_constraint_matrix = jnp.eye(2)
    fitter._eline_amplitude_names = ["eline_amp_Halpha", "eline_amp_Hbeta"]
    fitter.model.wave_obs = jnp.linspace(4000.0, 8000.0, 3)
    fitter.model._spectral_resolution = 2000.0
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, ELineFittedLikelihood)
    assert lk.channel == "spec_fnu"
    assert tuple(lk.amplitude_names) == ("eline_amp_Halpha", "eline_amp_Hbeta")


@pytest.mark.unit
def test_joint_without_n_data_phot_raises_assertion():
    """Joint data requires model.observation.n_data_phot — used to be a
    silent None bail-out (so legacy χ² fall-through fired with the
    wrong likelihood); now an explicit error flags the
    misconfiguration loudly. Phase II-2.3 cleanup."""
    fitter = _make_proxy(
        data=[1.0] * 6,
        noise=[0.1] * 6,
        data_type="joint",
        n_data_phot=None,
    )
    with pytest.raises(ValueError, match="n_data_phot"):
        Fitter._maybe_build_default_likelihood(fitter)


# ──────────────────────────────────────────────────────────────────────
# Phase II-2.3 — combined cal+eline + joint variable_noise
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_cal_marg_plus_eline_marg_auto_builds_combined_adapter():
    """Combined calibration + eline marginalisation → single adapter
    (Prospector-style galaxy spectroscopy fitting). Was a legacy
    fall-through pre-II-2.3."""
    from tengri.inference.likelihoods import CalibrationELineMarginalisedLikelihood

    fitter = _make_proxy(
        data=[1.0, 2.0, 3.0],
        noise=[0.1, 0.1, 0.1],
        data_type="spectroscopy",
        has_cal=True,
        has_eline=True,
    )
    fitter._eline_wavelengths = jnp.array([6564.6, 4861.3])
    fitter._eline_independent_wavelengths = jnp.array([6564.6, 4861.3])
    fitter._eline_constraint_matrix = jnp.eye(2)
    fitter._eline_prior_sigma = 1e10
    fitter._eline_prior_width_dex = 0.5
    fitter.model.wave_obs = jnp.linspace(4000.0, 8000.0, 3)
    fitter.model._spectral_resolution = 2000.0
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, CalibrationELineMarginalisedLikelihood)
    assert lk.eline_prior_type == "flat"
    assert lk.channel == "spec_fnu"


@pytest.mark.unit
def test_cal_marg_plus_eline_cloudy_auto_builds_combined_adapter_cloudy():
    """Cloudy variant of the combined cal + eline adapter."""
    from tengri.inference.likelihoods import CalibrationELineMarginalisedLikelihood

    fitter = _make_proxy(
        data=[1.0, 2.0, 3.0],
        noise=[0.1, 0.1, 0.1],
        data_type="spectroscopy",
        has_cal=True,
        has_eline=True,
    )
    fitter._eline_prior_type = "cloudy"
    fitter._eline_prior_sigma = 1e10
    fitter._eline_wavelengths = jnp.array([6564.6, 4861.3])
    fitter._eline_independent_wavelengths = jnp.array([6564.6, 4861.3])
    fitter._eline_constraint_matrix = jnp.eye(2)
    fitter._eline_prior_width_dex = 0.5
    fitter.model.wave_obs = jnp.linspace(4000.0, 8000.0, 3)
    fitter.model._spectral_resolution = 2000.0
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, CalibrationELineMarginalisedLikelihood)
    assert lk.eline_prior_type == "cloudy"


@pytest.mark.unit
def test_cal_marg_plus_eline_fitted_raises_not_implemented():
    """cal_marg + eline_fitted is a legitimate-but-unsupported combo.
    Raise loudly rather than silently producing wrong likelihoods."""
    fitter = _make_proxy(
        data=[1.0, 2.0],
        noise=[0.1, 0.1],
        data_type="spectroscopy",
        has_cal=True,
    )
    fitter._eline_fitted = True
    with pytest.raises(NotImplementedError, match="calibration"):
        Fitter._maybe_build_default_likelihood(fitter)


@pytest.mark.unit
def test_spectroscopy_with_variable_noise_auto_builds_student_t():
    """Spec + variable_noise → StudentTLikelihood pinned to spec_fnu."""
    from tengri.inference.likelihoods import StudentTLikelihood

    fitter = _make_proxy(
        data=[1.0, 2.0, 3.0],
        noise=[0.1, 0.1, 0.1],
        data_type="spectroscopy",
    )
    # Mock has_noise_model returning True via spec.all_params having the right entry
    from unittest.mock import patch

    with patch("tengri.observation.noise.has_noise_model", return_value=True):
        lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, StudentTLikelihood)
    assert lk.channel == "spec_fnu"


@pytest.mark.unit
def test_joint_with_variable_noise_auto_builds_composite_of_student_t():
    """Joint + variable_noise → CompositeLikelihood of two Student-t
    (one per channel, sharing f_cal_param='noise_frac_cal'). Was a
    legacy fall-through pre-II-2.3."""
    from unittest.mock import patch

    from tengri.inference.likelihoods import StudentTLikelihood

    fitter = _make_proxy(
        data=[1.0, 2.0, 3.0, 4.0, 5.0],
        noise=[0.1] * 5,
        data_type="joint",
        n_data_phot=2,
    )
    with patch("tengri.observation.noise.has_noise_model", return_value=True):
        lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, CompositeLikelihood)
    assert len(lk.likelihoods) == 2
    phot_lk, spec_lk = lk.likelihoods
    assert isinstance(phot_lk, StudentTLikelihood)
    assert isinstance(spec_lk, StudentTLikelihood)
    assert phot_lk.channel == "phot_fnu"
    assert spec_lk.channel == "spec_fnu"
