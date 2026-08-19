# SPDX-License-Identifier: BSD-3-Clause
"""A typo'd constructor-routed fit option must be correctable (#1469 follow-up).

``check_unknown_kwargs`` builds its suggestion pool from the runner's
signature. Constructor-routed options -- the ``Fitter.__init__`` parameters
that :func:`~tengri.inference.fitter.split_fitter_kwargs` sends to
construction rather than to the runner -- are documented ``fit()`` kwargs but
are not runner parameters, so they were absent from the pool.

The result: ``fit(..., calibration_marginalze=True)`` was rejected with a list
of accepted names that did not contain ``calibration_marginalize``, and no
correction. The message was complete and still unhelpful, because the one name
the user wanted was the one it could not mention.

The rejection set is deliberately NOT widened. A correctly spelled constructor
kwarg is routed away by ``split_fitter_kwargs`` and never reaches this check,
so accepting one here would only let it fail deeper in.

The exception stays ``TypeError`` (#1629): the type is what callers catch, and
the same mistake raises ``TypeError`` from Python itself and from
``SEDModel.build``. This is a message fix, not a type change.
"""

from __future__ import annotations

import inspect

import pytest

from tengri.inference._backend_registry import check_unknown_kwargs, lookup_backend
from tengri.inference.fitter import _FIT_SURFACE_POSITIONAL, Fitter

pytestmark = pytest.mark.regression_bug


def _ctor_names() -> set[str]:
    return {
        name
        for name in inspect.signature(Fitter.__init__).parameters
        if name not in _FIT_SURFACE_POSITIONAL
    }


def test_typod_constructor_option_is_suggested():
    """The correction must name the real option, not just reject the typo."""
    entry = lookup_backend("map")

    with pytest.raises(TypeError) as excinfo:
        check_unknown_kwargs(
            entry, {"calibration_marginalze": True}, also_accepted=frozenset(_ctor_names())
        )

    message = str(excinfo.value)
    assert "calibration_marginalze" in message, f"the typo is not named: {message}"
    assert "calibration_marginalize" in message, (
        "the intended spelling is not suggested; the suggestion pool is the "
        f"runner signature alone, which cannot contain it: {message}"
    )


def test_suggestion_pool_does_not_widen_the_rejection():
    """A constructor name must still be refused by the runner check.

    Suggestions and acceptance are different questions. A correctly spelled
    constructor kwarg never reaches here in practice, but if one did, letting
    it through would just move the failure deeper into the backend.
    """
    entry = lookup_backend("map")

    with pytest.raises(TypeError):
        check_unknown_kwargs(
            entry,
            {"calibration_marginalize": True},
            also_accepted=frozenset(_ctor_names()),
        )


def test_exception_type_is_unchanged():
    """#1629 made TypeError the contract; this change must not move it."""
    entry = lookup_backend("map")

    with pytest.raises(TypeError):
        check_unknown_kwargs(entry, {"definitely_not_a_kwarg": 1})


def test_runner_options_are_still_suggested():
    """The original pool must survive the addition."""
    entry = lookup_backend("map")

    with pytest.raises(TypeError) as excinfo:
        check_unknown_kwargs(entry, {"n_stps": 10}, also_accepted=frozenset(_ctor_names()))

    assert "n_steps" in str(excinfo.value), str(excinfo.value)


def test_end_to_end_through_the_fit_surface(synthetic_ssp_wide, synthetic_tophat_obs):
    """The suggestion must reach a real ``fit()`` caller, not just the helper.

    Checking the helper alone would pass even if ``Fitter.run`` never passed
    the constructor names in.
    """
    import jax
    import numpy as np

    from tengri import FIXED, FREE, Fixed, SEDModel

    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "*": FREE},
        dust={"law_diff": "calzetti", "type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    truth = model.spec.sample(jax.random.PRNGKey(0))
    flux = np.asarray(model.predict(truth).photometry())
    noise = np.abs(flux) * 0.05 + 1e-30

    with pytest.raises(TypeError) as excinfo:
        model.fit(data=flux, noise=noise, method="map", n_steps=2, calibration_marginalze=True)

    assert "calibration_marginalize" in str(excinfo.value), str(excinfo.value)
