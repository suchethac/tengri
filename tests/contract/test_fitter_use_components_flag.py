from __future__ import annotations

# SPDX-License-Identifier: BSD-3-Clause
import pytest

pytestmark = pytest.mark.contract

"""Phase II opt-in: Fitter(use_components=True) routes prediction through
the orchestrator seam (:meth:`SEDModel._photometry_via_state`; the public
``predict_photometry_components`` name is a deprecation shim since cleanup
PR-2) instead of the legacy fused kernel.

Smoke-tests only — full numerical-equivalence sweep is part of the eventual
Phase B cutover. Here we verify (1) the flag is honored, (2) wiring picks
the orchestrator bridge in :func:`_build_prediction`, and (3) the
non-photometry guard rejects the unsupported configuration loudly.
"""


from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from tengri.inference.loss_functions import build_loglikelihood_fn
from tengri.inference.photometry_likelihood import PhotometryLikelihood
from tests._shared_mocks import MockSpec

_MockSpec = MockSpec


class _DualPathModel:
    """Model exposing both legacy and orchestrator photometry/spectrum paths.

    Each path returns a tagged sentinel so the test can read which one
    the loss function called.
    """

    def __init__(self):
        self.spec = None
        self.legacy_calls = 0
        self.orchestrator_calls = 0
        self._wave_obs = jnp.array([4000.0, 5000.0, 6000.0])

    def predict_photometry(self, params):
        self.legacy_calls += 1
        return jnp.array([1.0, 2.0, 3.0])

    def _photometry_via_state(self, params):
        self.orchestrator_calls += 1
        return jnp.array([10.0, 20.0, 30.0])

    def predict_spectrum(self, params, wave_obs=None):
        self.legacy_calls += 1
        return jnp.array([1.0, 2.0, 3.0])

    def _spectrum_via_state(self, params, wave_obs=None):
        self.orchestrator_calls += 1
        return jnp.array([10.0, 20.0, 30.0])


def _make_fitter(*, use_components: bool):
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
        use_components=use_components,
    )
    return fitter, model


def test_default_routes_through_legacy_path():
    fitter, model = _make_fitter(use_components=False)
    loglik = build_loglikelihood_fn(fitter)
    _ = loglik({"flux_scale": 1.0}, fitter._data_args)
    # build_loglikelihood_fn calls predict once during channel-scale pre-check,
    # then the returned function calls predict once during evaluation (total: 2).
    assert model.legacy_calls == 2
    assert model.orchestrator_calls == 0


def test_use_components_routes_through_component_path():
    fitter, model = _make_fitter(use_components=True)
    loglik = build_loglikelihood_fn(fitter)
    _ = loglik({"flux_scale": 1.0}, fitter._data_args)
    # build_loglikelihood_fn calls predict once during channel-scale pre-check,
    # then the returned function calls predict once during evaluation (total: 2).
    assert model.legacy_calls == 0
    assert model.orchestrator_calls == 2


def _make_spectroscopy_fitter(*, use_components: bool):
    spec = _MockSpec(free_names=["flux_scale"])
    model = _DualPathModel()
    model.spec = spec

    fnu_obs = jnp.array([1.0, 2.0, 3.0])
    fnu_err = jnp.array([0.1, 0.1, 0.1])

    fitter = SimpleNamespace(
        model=model,
        data=fnu_obs,
        noise=fnu_err,
        data_type="spectroscopy",
        data_mask=None,
        spec=spec,
        _free_names=spec.free_params,
        _fixed_values={},
        _bounds={"flux_scale": (-jnp.inf, jnp.inf)},
        _data_args={"data": fnu_obs, "noise": fnu_err},
        _user_likelihood=None,  # use legacy χ² path; data type drives routing
        use_components=use_components,
    )
    return fitter, model


def test_use_components_routes_spectrum_through_component_path():
    """Spectroscopy now also routes through the orchestrator when opted in."""
    fitter, model = _make_spectroscopy_fitter(use_components=True)
    # We're not asserting numerical outcome (no user_likelihood / loss math
    # here) — just that the dispatch in _build_prediction picks the
    # orchestrator branch.
    from tengri.inference.loss_functions import _build_prediction

    _build_prediction(
        model,
        {"flux_scale": 1.0},
        "spectroscopy",
        has_line_fluxes=False,
        has_indices=False,
        index_defs=None,
        data_args=fitter._data_args,
        use_components=True,
    )
    assert model.legacy_calls == 0
    assert model.orchestrator_calls == 1


def test_use_components_rejects_unknown_data_type_at_construction():
    """Sanity: an unknown data_type with use_components=True still raises."""
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
            data_type="not_a_real_type",
            use_components=True,
            auto_protocol_likelihood=False,
        )
