# SPDX-License-Identifier: BSD-3-Clause
"""The MAP certificate runs only when scipy's stop needs checking (#3530).

``_certify_map_point`` re-evaluates the gradient at scipy's point and, for a converged
fit, returns at once. The dispatch must not call it at all in that case: on a warm
quickstart fit the call cost about 0.6 s of compile-cache reload for a gradient that
was already known to be small. The contract pinned here:

* scipy converged (success, max |grad| < gtol): ``_certify_map_point`` is not called;
* scipy reported failure, or the reported point has max |grad| >= gtol:
  ``_certify_map_point`` is called, and the fit is still certified as before.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri import Fixed
from tengri.inference.backends import map_dispatch
from tengri.inference.fitter import Fitter
from tests.contract.test_batched_map_lbfgs import _dpl_template
from tests.contract.test_batched_spectro_loglik import _mock_catalog
from tests.contract.test_batched_spectrum_forward import _banded

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def galaxy(synthetic_ssp):
    g = _mock_catalog(synthetic_ssp)[0]
    template = _dpl_template(synthetic_ssp, g["wave"], _banded(g["wave"]), redshift=Fixed(g["z"]))
    return template, g


def _fitter(template, g):
    return Fitter(
        template,
        data=jnp.asarray(g["flux"]),
        noise=jnp.asarray(g["sigma"]),
        data_type="spectroscopy",
        approx=None,
        profile_mass=False,
    )


@pytest.fixture
def certify_calls(monkeypatch):
    """Count calls to the certificate, leaving its behaviour unchanged."""
    real = map_dispatch._certify_map_point
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(map_dispatch, "_certify_map_point", counted)
    return calls


def _patch_scipy_result(monkeypatch, *, success=None, shift=None):
    """Wrap scipy's minimize so its reported success or point is changed, not computed."""
    import scipy.optimize as so

    real = so.minimize

    def wrapped(*args, **kwargs):
        result = real(*args, **kwargs)
        if success is not None:
            result.success = success
        if shift is not None:
            result.x = result.x + shift
        return result

    monkeypatch.setattr(so, "minimize", wrapped)


def test_converged_fit_does_not_call_the_certificate(galaxy, certify_calls):
    template, g = galaxy
    post = _fitter(template, g).run("map", tol=1e-3)
    assert post.diagnostics["map_converged"] is True
    assert post.diagnostics["map_decrement"] is None
    assert certify_calls == []


def test_scipy_failure_calls_the_certificate(galaxy, certify_calls, monkeypatch):
    template, g = galaxy
    _patch_scipy_result(monkeypatch, success=False)
    post = _fitter(template, g).run("map", tol=1e-3)
    assert certify_calls == [1]
    assert post.diagnostics["map_converged"] is True


def test_large_gradient_at_a_successful_stop_calls_the_certificate(
    galaxy, certify_calls, monkeypatch
):
    template, g = galaxy
    _patch_scipy_result(monkeypatch, success=True, shift=0.5)
    _fitter(template, g).run("map", tol=1e-3)
    assert certify_calls == [1]
