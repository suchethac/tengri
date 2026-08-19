# SPDX-License-Identifier: BSD-3-Clause
"""Regression: fit-surface kwargs that belong to the Fitter constructor must reach it (#1378).

Spec #1320 §7 teaches, marked ✓:

    fwd.fit(data, calibration_marginalize=True, cal_n_poly=3, ...)
    fwd.fit(..., likelihood=MyLikelihood())

but ``calibration_marginalize`` / ``cal_n_poly`` / ``likelihood`` (and friends) are
**Fitter constructor** parameters, while every convenience surface routed its whole
``**kwargs`` to ``Fitter.run()`` — the taught calls died with a ``TypeError`` inside
the backend. Same class on the sed surface: the WavePrecomp docstring teaches
``model.fit(row.data, params={"redshift": row.z})`` and ``params`` never reached
``params_override`` on that path (#1329's fix covered ``forward.fit`` only).

The routing allowlist is derived from the live ``Fitter.__init__`` signature so it
cannot drift when the constructor gains parameters.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import FIXED, SEDModel, Uniform
from tengri.forward.forward_model import ForwardModel
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={
            "law_diff": "calzetti",
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
        },
        neb={"type": "none"},
        redshift=FIXED,
    )


@pytest.fixture(scope="module")
def mock_data(model):
    params = model.spec.sample(jax.random.PRNGKey(0))
    flux = np.asarray(model.predict_photometry(params))
    return flux, 0.05 * np.abs(flux)


@pytest.fixture()
def ctor_spy(monkeypatch):
    """Record every kwarg Fitter.__init__ receives, then construct normally.

    ``functools.wraps`` keeps the spy signature-transparent: the #1378 routing
    derives its allowlist from ``inspect.signature(Fitter.__init__)``, which
    follows ``__wrapped__`` — a bare ``(*args, **kwargs)`` spy would silently
    empty the allowlist and defeat the very thing under test.
    """
    import functools

    seen = []
    original = Fitter.__init__

    @functools.wraps(original)
    def _spy(self, *args, **kwargs):
        seen.append(dict(kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Fitter, "__init__", _spy)
    return seen


def test_fwd_fit_routes_calibration_flags_to_the_constructor(model, mock_data, ctor_spy):
    """The spec §7 headline call must run, with the flags landing at construction."""
    flux, err = mock_data
    fwd = ForwardModel.build(sed=model)

    fwd.fit(flux, err, method="map", n_steps=3, calibration_marginalize=True, cal_n_poly=4)

    assert ctor_spy, "Fitter was never constructed"
    received = ctor_spy[-1]
    assert received.get("calibration_marginalize") is True
    assert received.get("cal_n_poly") == 4


def test_sed_fit_routes_ctor_kwargs(model, mock_data, ctor_spy):
    """Same routing on the sed.fit sugar (fit_model path)."""
    flux, err = mock_data

    model.fit(flux, err, method="map", n_steps=3, calibration_marginalize=True)

    assert ctor_spy, "Fitter was never constructed"
    assert ctor_spy[-1].get("calibration_marginalize") is True


def test_sed_fit_params_reaches_params_override(model, mock_data):
    """The WavePrecomp docstring's taught call: model.fit(..., params={...}).

    #1329's plumbing covered forward.fit only; the sed surface raised
    ``TypeError: run_map() got an unexpected keyword argument 'params'``.
    """
    flux, err = mock_data

    post = model.fit(flux, err, method="map", n_steps=3, params={"met_logzsol": -0.5})

    assert abs(float(post.params["met_logzsol"]) - (-0.5)) < 1e-12


def test_unknown_kwargs_still_fail_loudly(model, mock_data):
    """Routing must not create a silent kwarg sink — typos still raise, by name.

    The rule is "loud", not "``TypeError``". Which exception surfaces depends on
    *where* the name is caught, and that moved twice in one day: Python itself
    rejected it inside the runner (``run_map() got an unexpected keyword
    argument``), then #1605 added a pre-dispatch check in ``_backend_registry``
    raising ``ValueError``, then #1629 settled it back to ``TypeError`` so the
    same mistake fails the same way from either layer. ``check_capabilities``
    still answers ``ValueError``, for *declared capability* names only.

    Pinning the type pinned the layer, so an improvement to the error broke this
    test while the guarded behavior was intact. Assert the invariant instead:
    it raises, and the message shows the pre-dispatch guard is what caught it.

    Match on ``does not accept``, not on the kwarg name. The name alone does not
    discriminate — **both** layers put it in the message, the runner's as
    ``run_map() got an unexpected keyword argument 'calibration_marginalze'``.
    With ``check_fit_kwargs`` neutered to a silent return, the kwarg falls
    through to the runner and dies at ``fitter.py`` with that ``TypeError``, so
    a name-only match (or a bare ``pytest.raises``) passes with the guard
    deleted — green on the exact regression this test exists to catch. Measured
    on ``main`` at 04231b02e: mutation applied, test still passed.

    ``does not accept`` appears only in the registry's message, so it separates
    "refused up front" from "died inside", which is the distinction #1378 is
    about. The type stays unpinned — the layer may move again.
    """
    flux, err = mock_data
    fwd = ForwardModel.build(sed=model)

    with pytest.raises((TypeError, ValueError), match=r"does not accept"):
        fwd.fit(flux, err, method="map", n_steps=3, calibration_marginalze=True)
