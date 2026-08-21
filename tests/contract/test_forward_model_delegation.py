# SPDX-License-Identifier: BSD-3-Clause
"""ForwardModel must support every observable configuration the inference stack uses.

``ForwardModel`` is the canonical inference surface (#211), but it has no
``__getattr__`` fall-through — it forwards to the inner SED through an explicit
list. The inference stack does ``model = fitter.model`` and then calls SEDModel
methods on it directly, so any method missing from that list raises
``AttributeError`` *at fit time*, for whichever observable configuration happens
to reach it.

That produced #1300: a line-flux fit through ``ForwardModel.fit`` died on
``_has_line_catalog``, then on ``measure_line_fluxes``, while the DEPRECATED
``Fitter(sed_model, ...)`` path worked fine. The recommended API was the broken
one. Nothing caught it because every existing line-flux test drove the
deprecated surface, and the notebook that used this configuration silenced the
deprecation warning that would have pointed at the mismatch.

Two guards here:

1. a static check that the declared delegation list is actually installed and
   still covers what ``src/tengri/inference`` calls on ``model``, and
2. a fit-level check per observable configuration, which is what makes a *new*
   configuration fail loudly rather than at some user's first run.
"""

from __future__ import annotations

import re
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.contract

from tengri import (
    FREE,
    Fixed,
    ForwardModel,
    NoiseModel,
    Observation,
    Photometry,
    SEDModel,
    builders,
    load_ssp_data,
)
from tengri.forward.sed_model import SEDModel as _SEDModelCls
from tengri.observation import LineFluxData

_SSP = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
_INFERENCE = Path(__file__).resolve().parents[2] / "src" / "tengri" / "inference"
_MODEL_ATTR = re.compile(
    r"(?:self\.model|model|fitter\.model|self\._model|context\.model)\.([a-zA-Z_][a-zA-Z0-9_]*)"
)


def test_declared_delegations_are_installed():
    """Every name in ``_DELEGATED_TO_INNER_SED`` must actually exist on the class.

    The list is the documentation; this makes it also the contract, so a name can
    not be added to the tuple without being installed (or removed from the class
    without being removed from the tuple).
    """
    for name in ForwardModel._DELEGATED_TO_INNER_SED:
        assert hasattr(ForwardModel, name), (
            f"{name!r} is declared in _DELEGATED_TO_INNER_SED but not installed on ForwardModel"
        )


def test_inference_stack_needs_nothing_ForwardModel_lacks():
    """Nothing the inference package calls on ``model`` may be missing from ForwardModel.

    This is the check that generalizes: it re-derives the required surface from
    the source rather than from a hand-maintained list, so a newly-added
    ``model.<something>`` call in inference/ fails here instead of at fit time.
    """
    needed = set()
    for path in _INFERENCE.rglob("*.py"):
        needed |= set(_MODEL_ATTR.findall(path.read_text()))

    missing = sorted(
        n for n in needed if hasattr(_SEDModelCls, n) and not hasattr(ForwardModel, n)
    )
    assert not missing, (
        "the inference stack calls these on `model`, SEDModel has them and ForwardModel does "
        f"not — a fit through the canonical surface will raise AttributeError: {missing}"
    )


def _model(ssp_data, observation):
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=observation,
        sfh={"type": ["dpl", "field"], "all_params": FREE},
        met={"logzsol": Fixed(-0.3)},
        dust_attenuation=builders.dust.two_component(defaults=FREE, law="calzetti"),
        neb=builders.neb.ssp(),
        redshift=Fixed(0.1),
        igm={"type": "none"},
        n_grid=8,
        approx=None,
    )


def _observation(kind):
    phot = Photometry.from_names(["galex_fuv", "sdss_g", "sdss_r", "2mass_ks"])
    noise = NoiseModel(calibration_floor=0.01, student_t_dof=None)
    if kind == "photometry":
        return Observation(photometry=phot, noise=noise)
    if kind == "line_fluxes":
        names = ("Halpha", "Hbeta", "OIII_5007")
        lines = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in names})
        return Observation(photometry=phot, line_fluxes=lines, noise=noise)
    raise AssertionError(f"unknown observable configuration {kind!r}")


@pytest.mark.skipif(not _SSP.exists(), reason="wNE SSP grid not available")
@pytest.mark.parametrize("kind", ["photometry", "line_fluxes"])
def test_fit_runs_through_the_canonical_surface(kind):
    """A short MAP fit must complete through ``ForwardModel.fit`` for each configuration.

    Deliberately end-to-end rather than a method-presence check: the failures in
    #1300 surfaced one at a time, each only once the previous was fixed, because
    they sit on different branches of the likelihood builder. Only running the
    fit exercises the branch a configuration actually takes.
    """
    ssp = load_ssp_data(str(_SSP))
    observation = _observation(kind)
    sed = _model(ssp, observation)
    forward = ForwardModel.build(sed=sed, observation=observation)

    params = {**sed.spec.get_fixed_values(), **sed.spec.sample(jax.random.PRNGKey(0))}
    mock = sed.mock(params, snr=20.0, key=jax.random.PRNGKey(1))

    res = forward.fit(
        np.asarray(mock.flux_obs),
        np.asarray(mock.noise),
        method="map",
        approx=None,
        n_steps=40,
        n_restarts=1,
        key=jax.random.PRNGKey(2),
        verbose=False,
    )
    assert res.params, f"{kind}: fit through ForwardModel returned no parameters"
    assert all(np.all(np.isfinite(np.asarray(v))) for v in res.params.values())


@pytest.mark.skipif(not _SSP.exists(), reason="wNE SSP grid not available")
def test_delegation_refuses_multi_population_rather_than_guessing():
    """Multi-population forwards must raise, not answer for ``populations[0]``.

    ``_inner_sed_for_delegation`` silently returns the first population. For a
    genuine multi-population decomposition there is no single right answer, and
    quietly using population 0 would be the same fail-open shape as #1271.
    """
    ssp = load_ssp_data(str(_SSP))
    observation = _observation("photometry")
    sed = _model(ssp, observation)
    forward = ForwardModel.build(sed=sed, observation=observation)

    two_pop = ForwardModel(
        populations=forward.populations + forward.populations, observation=observation
    )
    with pytest.raises(NotImplementedError, match="populations"):
        two_pop._has_line_catalog()
