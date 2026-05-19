# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the Phase II-1 :class:`Likelihood` adapter cohort.

Three adapters under test:

- :class:`PhotometryLikelihood` — diagonal Gaussian over filter fluxes
  (``prediction["phot_fnu"]``).
- :class:`SpectroscopyLikelihood` — diagonal Gaussian over spectrum
  pixels (``prediction["spec_fnu"]``).
- :class:`CompositeLikelihood` — sum of constituents.

Each is checked for: contract conformance, numerical correctness vs the
shared :func:`diag_gaussian_log_prob` helper, ``sigma_floor`` behaviour,
and (for the composite) commutativity, no-side-effect summation, and
duplicate-parameter detection.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.protocols import Likelihood
from tengri.inference.composite_likelihood import CompositeLikelihood
from tengri.inference.likelihoods.gaussian import (
    diag_gaussian_chi2,
    diag_gaussian_log_prob,
)
from tengri.inference.photometry_likelihood import PhotometryLikelihood
from tengri.inference.spectroscopy_likelihood import SpectroscopyLikelihood

# ──────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def phot_data():
    return {
        "fnu_obs": jnp.array([1.1e-29, 1.9e-29, 3.05e-29]),
        "fnu_err": jnp.array([0.1e-29, 0.1e-29, 0.1e-29]),
        "fnu_pred": jnp.array([1.0e-29, 2.0e-29, 3.0e-29]),
    }


@pytest.fixture
def spec_data():
    rng = jnp.linspace(1.0, 2.0, 32)
    return {
        "fnu_obs": rng,
        "fnu_err": jnp.ones_like(rng) * 0.05,
        "fnu_pred": rng + 0.02,  # constant 0.02 offset
    }


# ──────────────────────────────────────────────────────────────────────
# Protocol conformance + numerical correctness
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "constructor, channel",
    [
        (PhotometryLikelihood, "phot_fnu"),
        (SpectroscopyLikelihood, "spec_fnu"),
    ],
)
def test_protocol_conformance(constructor, channel):
    """Each adapter duck-types as :class:`Likelihood`."""
    lk = constructor(fnu_obs=jnp.zeros(4), fnu_err=jnp.ones(4))
    assert isinstance(lk, Likelihood)
    assert lk.declared_parameters() == []
    assert isinstance(lk.name, str) and channel.split("_")[0] in lk.name


@pytest.mark.unit
def test_photometry_likelihood_matches_helper(phot_data):
    """PhotometryLikelihood.log_prob == diag_gaussian_log_prob (no floor)."""
    lk = PhotometryLikelihood(phot_data["fnu_obs"], phot_data["fnu_err"])
    pred = {"phot_fnu": phot_data["fnu_pred"]}
    expected = diag_gaussian_log_prob(
        phot_data["fnu_pred"], phot_data["fnu_obs"], phot_data["fnu_err"]
    )
    assert float(lk.log_prob(pred)) == pytest.approx(float(expected), rel=1e-10)


@pytest.mark.unit
def test_spectroscopy_likelihood_matches_helper(spec_data):
    """SpectroscopyLikelihood.log_prob == diag_gaussian_log_prob."""
    lk = SpectroscopyLikelihood(spec_data["fnu_obs"], spec_data["fnu_err"])
    pred = {"spec_fnu": spec_data["fnu_pred"]}
    expected = diag_gaussian_log_prob(
        spec_data["fnu_pred"], spec_data["fnu_obs"], spec_data["fnu_err"]
    )
    assert float(lk.log_prob(pred)) == pytest.approx(float(expected), rel=1e-10)


@pytest.mark.unit
def test_sigma_floor_inflates_variance(phot_data):
    """Adding sigma_floor=0.1 makes χ² strictly smaller (variance grows)."""
    no_floor = PhotometryLikelihood(phot_data["fnu_obs"], phot_data["fnu_err"])
    with_floor = PhotometryLikelihood(phot_data["fnu_obs"], phot_data["fnu_err"], sigma_floor=0.1)
    pred = {"phot_fnu": phot_data["fnu_pred"]}
    chi2_no = float(
        diag_gaussian_chi2(phot_data["fnu_pred"], phot_data["fnu_obs"], phot_data["fnu_err"])
    )
    # log_prob is -0.5 * χ², so a smaller χ² means a larger log_prob.
    assert float(with_floor.log_prob(pred)) > float(no_floor.log_prob(pred))
    # Sanity: floor=0 reproduces the no-floor χ² exactly.
    assert -2 * float(no_floor.log_prob(pred)) == pytest.approx(chi2_no, rel=1e-10)


# ──────────────────────────────────────────────────────────────────────
# CompositeLikelihood
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_composite_sums_constituents(phot_data, spec_data):
    """CompositeLikelihood == sum of phot + spec log_probs."""
    phot_lk = PhotometryLikelihood(phot_data["fnu_obs"], phot_data["fnu_err"])
    spec_lk = SpectroscopyLikelihood(spec_data["fnu_obs"], spec_data["fnu_err"])
    composite = CompositeLikelihood(phot_lk, spec_lk)
    prediction = {
        "phot_fnu": phot_data["fnu_pred"],
        "spec_fnu": spec_data["fnu_pred"],
    }
    expected = phot_lk.log_prob(prediction) + spec_lk.log_prob(prediction)
    assert float(composite.log_prob(prediction)) == pytest.approx(float(expected), rel=1e-10)


@pytest.mark.unit
def test_composite_is_commutative(phot_data, spec_data):
    """Order of constituents must not affect the final scalar."""
    phot_lk = PhotometryLikelihood(phot_data["fnu_obs"], phot_data["fnu_err"])
    spec_lk = SpectroscopyLikelihood(spec_data["fnu_obs"], spec_data["fnu_err"])
    a = CompositeLikelihood(phot_lk, spec_lk)
    b = CompositeLikelihood(spec_lk, phot_lk)
    pred = {"phot_fnu": phot_data["fnu_pred"], "spec_fnu": spec_data["fnu_pred"]}
    assert float(a.log_prob(pred)) == pytest.approx(float(b.log_prob(pred)), rel=1e-10)


@pytest.mark.unit
def test_composite_diagnostic_name_lists_constituents(phot_data, spec_data):
    """Default ``name`` reflects ordering and constituent names."""
    composite = CompositeLikelihood(
        PhotometryLikelihood(phot_data["fnu_obs"], phot_data["fnu_err"]),
        SpectroscopyLikelihood(spec_data["fnu_obs"], spec_data["fnu_err"]),
    )
    assert "photometry_gaussian" in composite.name
    assert "spectroscopy_gaussian" in composite.name


@pytest.mark.unit
def test_composite_rejects_duplicate_declared_parameters():
    """Two likelihoods that own the same parameter name must raise."""
    from tengri.parameters.priors import Uniform

    class _DuplicateLk:
        name = "dup"

        def __init__(self):
            from tengri.protocols import ParamDeclaration

            self._decls = [ParamDeclaration("noise_jitter", Uniform(0.0, 1.0), "")]

        def log_prob(self, prediction, params):
            return jnp.asarray(0.0)

        def declared_parameters(self):
            return self._decls

    with pytest.raises(ValueError, match="declared by both"):
        CompositeLikelihood(_DuplicateLk(), _DuplicateLk())


@pytest.mark.unit
def test_composite_unions_declared_parameters():
    """Constituents' declared params are concatenated (no dedup needed when
    names differ)."""
    from tengri.protocols import ParamDeclaration
    from tengri.parameters.priors import Uniform

    class _A:
        name = "a"

        def log_prob(self, p, par):
            return jnp.asarray(0.0)

        def declared_parameters(self):
            return [ParamDeclaration("noise_a", Uniform(0.0, 1.0), "a")]

    class _B:
        name = "b"

        def log_prob(self, p, par):
            return jnp.asarray(0.0)

        def declared_parameters(self):
            return [ParamDeclaration("noise_b", Uniform(0.0, 1.0), "b")]

    composite = CompositeLikelihood(_A(), _B())
    decls = composite.declared_parameters()
    names = [d.name for d in decls]
    assert names == ["noise_a", "noise_b"]


@pytest.mark.unit
def test_composite_with_phot_only_ignores_spec_in_prediction(phot_data, spec_data):
    """A constituent reads only its own keys — extras are harmless."""
    phot_lk = PhotometryLikelihood(phot_data["fnu_obs"], phot_data["fnu_err"])
    composite = CompositeLikelihood(phot_lk)
    pred = {"phot_fnu": phot_data["fnu_pred"], "spec_fnu": spec_data["fnu_pred"]}
    expected = phot_lk.log_prob(pred)
    assert float(composite.log_prob(pred)) == pytest.approx(float(expected), rel=1e-10)
