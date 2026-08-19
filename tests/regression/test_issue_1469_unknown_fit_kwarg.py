# SPDX-License-Identifier: BSD-3-Clause
"""An unknown fit kwarg must name the method, not a backend internal (#1469).

``Catalog.fit(**kwargs)`` and ``ForwardModel.fit(**kwargs)`` both forward
unrecognized names to the registered runner. Runners like ``run_map`` take no
``**kwargs``, so a typo or an unsupported channel surfaced as::

    TypeError: run_map() got an unexpected keyword argument 'lines'

naming a function the caller never mentioned, from inside a backend they did
not choose. :func:`~tengri.inference._backend_registry.check_capabilities`
already fixed this for *declared capability* kwargs (``precondition=``); this
is the same answer for every other unknown name.

The check lives at the dispatch seam rather than on ``Catalog``, so both fit
surfaces are covered by one rule.
"""

from __future__ import annotations

import jax
import pytest

pytestmark = pytest.mark.regression_bug


def _unknown_kwarg_error(fn, **kwargs):
    """Run ``fn`` expecting a rejection; return the raised exception.

    Catches both types on purpose. The type of this rejection has moved twice
    in a day — Python's own ``TypeError`` from inside the runner, then
    ``ValueError`` when #1605 added the pre-dispatch check, then ``TypeError``
    again when #1629 settled it — so pinning it pins the layer, not the rule
    (#1636). The tests below assert on the *message* instead: ``does not
    accept`` appears only in the registry's rejection, so it distinguishes
    "refused up front" from "died inside the backend", which is the whole of
    #1469. Without that, a test passes with the guard deleted.
    """
    with pytest.raises((ValueError, TypeError)) as excinfo:
        fn(**kwargs)
    return excinfo.value


def test_catalog_fit_names_the_method_not_the_runner(synthetic_ssp_wide, synthetic_tophat_obs):
    """``cat.fit(lines=...)`` must not leak ``run_map``."""
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    cat, _ = build_two_galaxy_catalog(
        halpha=(1.0, 4.0), ssp=synthetic_ssp_wide, obs_base=synthetic_tophat_obs
    )

    err = _unknown_kwarg_error(
        cat.fit, method="map", key=jax.random.PRNGKey(0), n_steps=2, lines={"Halpha": (1.0, 0.1)}
    )
    msg = str(err)

    assert "does not accept" in msg, (
        f"the pre-dispatch guard is not what caught this — the kwarg fell through "
        f"to the runner, which is the #1469 regression itself: {msg}"
    )
    assert "run_map" not in msg, (
        f"the error leaks the backend function name the caller never mentioned: {msg}"
    )
    assert "lines" in msg, f"the error does not name the offending argument: {msg}"
    assert "map" in msg, f"the error does not name the method that rejected it: {msg}"


def test_error_lists_arguments_the_method_does_accept(synthetic_ssp_wide, synthetic_tophat_obs):
    """The message must show a way forward, and the advice must be real.

    An error that recommends something its own caller refuses is #1576. The
    accepted names are read off the live runner signature, so they cannot
    drift from what the backend takes.
    """
    from tengri.inference._backend_registry import lookup_backend
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    cat, _ = build_two_galaxy_catalog(
        halpha=(1.0, 4.0), ssp=synthetic_ssp_wide, obs_base=synthetic_tophat_obs
    )

    msg = str(
        _unknown_kwarg_error(
            cat.fit, method="map", key=jax.random.PRNGKey(0), n_steps=2, nonsense_kwarg=3
        )
    )

    entry = lookup_backend("map")
    import inspect

    accepted = {
        name
        for name, p in inspect.signature(entry.runner).parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    } - {"context", "fitter", "self", "key", "init_from"}

    named = [a for a in accepted if a in msg]
    assert named, (
        f"the error names no argument 'map' actually accepts (of {sorted(accepted)}): {msg}"
    )


def test_a_supported_kwarg_still_reaches_the_backend(synthetic_ssp_wide, synthetic_tophat_obs):
    """The guard must not reject arguments the runner does take.

    ``n_steps`` is a real ``run_map`` parameter; a check that rejected it
    would be worse than the leak it replaces.
    """
    from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

    cat, _ = build_two_galaxy_catalog(
        halpha=(1.0, 4.0), ssp=synthetic_ssp_wide, obs_base=synthetic_tophat_obs
    )

    post = cat.fit(method="map", key=jax.random.PRNGKey(0), n_steps=3)
    assert post.n_galaxies == 2


def test_single_galaxy_surface_gets_the_same_answer(synthetic_ssp_wide, synthetic_tophat_obs):
    """The rule lives at dispatch, so ``SEDModel.fit`` is covered too.

    Fixing this only on ``Catalog`` would leave the identical leak on the
    surface most users reach first.
    """
    import numpy as np

    from tengri import FIXED, FREE, Fixed, SEDModel

    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "*": FREE},
        dust={"law_diff": 'calzetti', "type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    truth = model.spec.sample(jax.random.PRNGKey(0))
    flux = np.asarray(model.predict(truth).photometry())
    noise = np.abs(flux) * 0.05 + 1e-30

    err = _unknown_kwarg_error(
        model.fit, data=flux, noise=noise, method="map", n_steps=2, nonsense_kwarg=3
    )

    assert "does not accept" in str(err), f"the guard did not catch it: {err}"
    assert "run_map" not in str(err), str(err)
