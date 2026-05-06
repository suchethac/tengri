# SPDX-License-Identifier: BSD-3-Clause
"""Phase II opt-in: Fitter(use_orchestrator=True) routes prediction through
:meth:`SEDModel.predict_photometry_via_orchestrator` instead of the legacy
fused kernel.

Smoke-tests only — full numerical-equivalence sweep is part of the eventual
Phase B cutover. Here we verify (1) the flag is honoured, (2) wiring picks
the orchestrator bridge in :func:`_build_prediction`, and (3) the
non-photometry guard rejects the unsupported configuration loudly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import jax.numpy as jnp
import pytest

from tengri.inference.loss_functions import build_loglikelihood_fn
from tengri.inference.photometry_likelihood import PhotometryLikelihood


class _IdentityDist:
    bounds = (-jnp.inf, jnp.inf)

    def unstandardize(self, x):
        return x


class _MockSpec:
    stochastic = False
    all_params: ClassVar[list] = []

    def __init__(self, free_names):
        self._free_names = free_names

    @property
    def free_params(self):
        return self._free_names

    def get_distribution(self, name):
        return _IdentityDist()

    def get_fixed_values(self):
        return {}

    def resolve_mirrors(self, params):
        return params


class _DualPathModel:
    """Model exposing both legacy and orchestrator photometry paths.

    Each path returns a tagged sentinel so the test can read which one
    the loss function called.
    """

    def __init__(self):
        self.spec = None
        self.legacy_calls = 0
        self.orchestrator_calls = 0

    def predict_photometry(self, params, mode=None):
        self.legacy_calls += 1
        return jnp.array([1.0, 2.0, 3.0])

    def predict_photometry_via_orchestrator(self, params):
        self.orchestrator_calls += 1
        return jnp.array([10.0, 20.0, 30.0])


def _make_fitter(*, use_orchestrator: bool):
    spec = _MockSpec(free_names=["flux_scale"])
    model = _DualPathModel()
    model.spec = spec

    fnu_obs = jnp.array([1.0, 2.0, 3.0])
    fnu_err = jnp.array([0.1, 0.1, 0.1])

    user_likelihood = PhotometryLikelihood(fnu_obs=fnu_obs, fnu_err=fnu_err)

    fitter = SimpleNamespace(
        model=model,
        data=fnu_obs,
        noise=fnu_err,
        data_type="photometry",
        data_mask=None,
        spec=spec,
        _free_names=spec.free_params,
        _fixed_values={},
        _bounds={"flux_scale": (-jnp.inf, jnp.inf)},
        _data_args={"data": fnu_obs, "noise": fnu_err},
        _user_likelihood=user_likelihood,
        use_orchestrator=use_orchestrator,
    )
    return fitter, model


def test_default_routes_through_legacy_path():
    fitter, model = _make_fitter(use_orchestrator=False)
    loglik = build_loglikelihood_fn(fitter, mode="auto")
    _ = loglik({"flux_scale": 1.0}, fitter._data_args)
    assert model.legacy_calls == 1
    assert model.orchestrator_calls == 0


def test_use_orchestrator_routes_through_component_path():
    fitter, model = _make_fitter(use_orchestrator=True)
    loglik = build_loglikelihood_fn(fitter, mode="auto")
    _ = loglik({"flux_scale": 1.0}, fitter._data_args)
    assert model.legacy_calls == 0
    assert model.orchestrator_calls == 1


def test_use_orchestrator_rejects_non_photometry_at_construction():
    """Until spectroscopy bridges land, Fitter must refuse the combo."""
    from tengri.inference.fitter import Fitter

    class _StubModel:
        spec = _MockSpec(free_names=["flux_scale"])

        def __init__(self):
            self.observation = SimpleNamespace(
                photometry=None, spectroscopy=None, spectral_indices=None
            )

    with pytest.raises(NotImplementedError, match="data_type"):
        Fitter(
            model=_StubModel(),
            data=jnp.array([1.0, 2.0]),
            noise=jnp.array([0.1, 0.1]),
            data_type="spectroscopy",
            use_orchestrator=True,
            auto_protocol_likelihood=False,
        )
