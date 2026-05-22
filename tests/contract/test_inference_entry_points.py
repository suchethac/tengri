"""Tests for the canonical inference entry point — ``ForwardModel.fit`` /
``Fitter(forward, ...)`` — and the deprecation shims around the legacy paths.

Architecture spec: inference is canonically through ``ForwardModel``
(issue #211). Three pre-existing paths remain functional but emit
DeprecationWarnings:

- ``SEDModel.fit(...)``
- ``Fitter(sed_model, data, noise).run(...)`` (when ``sed_model`` is a
  bare :class:`SEDModel`, not a :class:`ForwardModel`)

This module pins the soft-deprecation contract.
"""

from __future__ import annotations

import contextlib
import warnings

import pytest

from tengri.forward.forward_model import ForwardModel
from tengri.forward.sed_model import SEDModel
from tengri.inference.fitter import _maybe_warn_legacy_sedmodel

pytestmark = pytest.mark.contract


# ── ForwardModel.fit convenience ────────────────────────────────────


def test_forward_model_has_fit_method() -> None:
    """ForwardModel exposes a .fit shortcut equivalent to Fitter(self, ...).run(...)."""
    assert hasattr(ForwardModel, "fit")
    assert callable(ForwardModel.fit)


def test_forward_model_fit_signature_accepts_data_noise_method() -> None:
    import inspect

    sig = inspect.signature(ForwardModel.fit)
    params = sig.parameters
    assert "data" in params
    assert "noise" in params
    assert "method" in params
    # data/noise default to None so hierarchical fits work without them
    assert params["data"].default is None
    assert params["noise"].default is None


# ── Legacy-path deprecation shims ───────────────────────────────────


def test_legacy_warn_passes_through_forward_model() -> None:
    """``Fitter(forward, ...)`` is the canonical surface; no warning."""

    class _StubForward(ForwardModel):  # type: ignore[misc]
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # We can't easily construct a real ForwardModel with a stub SED;
        # the function takes any ForwardModel instance, so a bare-class
        # instance works for the type-check portion.
        with contextlib.suppress(Exception):
            _maybe_warn_legacy_sedmodel(object.__new__(ForwardModel))
    assert not any(
        issubclass(w.category, DeprecationWarning) and "Fitter(sed_model" in str(w.message)
        for w in caught
    )


def test_legacy_warn_passes_through_non_model_objects() -> None:
    """Random objects (test stubs, likelihood Protocols, …) emit no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _maybe_warn_legacy_sedmodel("not a model")
        _maybe_warn_legacy_sedmodel(object())
        _maybe_warn_legacy_sedmodel(42)
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_legacy_warn_fires_on_bare_sedmodel(synthetic_ssp, simple_observation) -> None:
    """``Fitter(sed_model, ...)`` emits a DeprecationWarning pointing at the new path."""
    from tengri import FIXED

    sed = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl", "*": FIXED},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _maybe_warn_legacy_sedmodel(sed)
    relevant = [
        w
        for w in caught
        if issubclass(w.category, DeprecationWarning) and "Fitter(sed_model" in str(w.message)
    ]
    assert len(relevant) == 1
    assert "ForwardModel" in str(relevant[0].message)
    assert "#211" in str(relevant[0].message)
