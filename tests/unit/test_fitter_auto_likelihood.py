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

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from tengri.inference.composite_likelihood import CompositeLikelihood
from tengri.inference.fitter import Fitter
from tengri.inference.photometry_likelihood import PhotometryLikelihood
from tengri.inference.spectroscopy_likelihood import SpectroscopyLikelihood

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

    Methods on :class:`Fitter` (``_build_base_likelihood``,
    ``_build_likelihood_extras``, ``_n_phot_split``,
    ``_make_eline_design_builder``) are invoked via the unbound form
    so we don't need a real :class:`Fitter` instance (which spawns a
    background compile thread we don't want for unit tests).
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
    """Make the proxy and self-reference it so bound lambdas work."""
    proxy = _make_helper_only_fitter(**kw)
    # Re-bind helpers to the actual proxy (lambda closure captured the
    # local '_proxy' name, which doesn't exist there — fix it now).
    proxy._build_base_likelihood = lambda: Fitter._build_base_likelihood(proxy)
    proxy._build_likelihood_extras = lambda: Fitter._build_likelihood_extras(proxy)
    proxy._n_phot_split = lambda: Fitter._n_phot_split(proxy)
    proxy._make_eline_design_builder = lambda: Fitter._make_eline_design_builder(proxy)
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
    assert phot.fnu_obs.shape == (n_phot,)
    assert spec.fnu_obs.shape == (5,)


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
    fitter.model._wave_obs = jnp.array([5000.0, 6000.0])
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
    fitter.model._wave_obs = jnp.linspace(4000.0, 8000.0, 3)
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
    fitter.model._wave_obs = jnp.linspace(4000.0, 8000.0, 3)
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
    assert indices_lk.obs.shape == (1,)


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
    fitter.model._wave_obs = jnp.linspace(4000.0, 8000.0, 3)
    fitter.model._spectral_resolution = 2000.0
    lk = Fitter._maybe_build_default_likelihood(fitter)
    assert isinstance(lk, ELineFittedLikelihood)
    assert lk.channel == "spec_fnu"
    assert tuple(lk.amplitude_names) == ("eline_amp_Halpha", "eline_amp_Hbeta")


@pytest.mark.unit
def test_joint_without_n_data_phot_falls_back_to_legacy():
    """Joint without an explicit phot/spec split → can't build composite."""
    fitter = _make_proxy(
        data=[1.0] * 6,
        noise=[0.1] * 6,
        data_type="joint",
        n_data_phot=None,
    )
    assert Fitter._maybe_build_default_likelihood(fitter) is None
