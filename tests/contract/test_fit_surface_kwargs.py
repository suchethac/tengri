# SPDX-License-Identifier: BSD-3-Clause
"""Three silent-failure defects on documented surfaces (#1500).

* ``forward.fit(data_type=...)`` was rejected: ``data_type`` is a real
  ``Fitter.__init__`` parameter, but ``_FIT_SURFACE_MANAGED`` excluded it from
  ``ctor_names``, so a user value fell through to ``run()`` and ``run_map()``
  raised ``unexpected keyword argument``.
* ``measure_line_fluxes(params)`` silently measured a built-in DESI set of five
  lines instead of the ones the model's own ``Observation`` declares.
* ``csp_integration`` is accepted and validated but does not reach the SED.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.inference.fitter import (
    _FIT_SURFACE_DERIVED,
    _FIT_SURFACE_POSITIONAL,
    split_fitter_kwargs,
)

pytestmark = pytest.mark.contract


def test_data_type_reaches_the_constructor_not_run():
    """A user-supplied data_type must be a ctor kwarg.

    Landing it in ``run_kwargs`` is what produced
    ``run_map() got an unexpected keyword argument 'data_type'``.
    """
    ctor, run = split_fitter_kwargs({"data_type": "joint", "n_steps": 10})
    assert ctor.get("data_type") == "joint", "data_type leaked to run()"
    assert "data_type" not in run
    assert run == {"n_steps": 10}


@pytest.mark.parametrize("name", sorted(_FIT_SURFACE_DERIVED))
def test_surface_derived_kwargs_are_accepted_from_the_user(name):
    """The surface may DERIVE these, but an explicit value is still meaningful.

    Excluding them wholesale made every one of them a ``run()`` kwarg, i.e. a
    guaranteed TypeError from whichever backend runner received it.
    """
    ctor, run = split_fitter_kwargs({name: object()})
    assert name in ctor, f"{name} was routed to run() instead of the constructor"
    assert name not in run


@pytest.mark.parametrize("name", sorted(_FIT_SURFACE_POSITIONAL))
def test_positional_surface_kwargs_are_never_taken_from_kwargs(name):
    """model/data/noise/self come from the call signature, never **kwargs."""
    ctor, _run = split_fitter_kwargs({name: object()})
    assert name not in ctor


def test_positional_and_derived_are_disjoint():
    assert not (_FIT_SURFACE_POSITIONAL & _FIT_SURFACE_DERIVED)


# --------------------------------------------------------------- line defaults


def _obs_with_lines(names, waves):
    """Minimal stand-in for an Observation that declares line fluxes."""
    from tengri.observation import LineFluxData

    lfd = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in names})
    assert tuple(lfd.names) == tuple(names)
    return type("Obs", (), {"line_fluxes": lfd, "has_line_fluxes": True})()


def test_resolver_uses_the_observations_own_lines():
    """Omitting line_defs must not silently measure a DIFFERENT set of lines.

    The old default was a built-in five-line DESI list, so a model declaring
    eight returned five, in a different identity order -- a plausible float array
    of the wrong length that only failed downstream, if at all.
    """
    from tengri.observation.line_measurement import resolve_line_defs

    names = (
        "Halpha",
        "Hbeta",
        "OIII_5007",
        "OIII_4959",
        "SII_6717",
        "SII_6731",
        "OII_3726",
        "OII_3729",
    )
    defs = resolve_line_defs(None, _obs_with_lines(names, None))
    assert len(defs) == len(names), f"expected {len(names)} line defs, got {len(defs)}"
    assert tuple(d.name for d in defs) == names


def test_resolver_falls_back_to_desi_without_an_observation():
    """No declared lines is the only case where the built-in default applies."""
    from tengri.observation.line_measurement import DESI_LINES, resolve_line_defs

    assert resolve_line_defs(None, None) == tuple(DESI_LINES)


def test_resolver_keeps_an_explicit_argument():
    from tengri.observation.line_measurement import default_line_defs, resolve_line_defs

    explicit = default_line_defs(np.array([6564.61]), ("Halpha",))
    out = resolve_line_defs(explicit, _obs_with_lines(("Halpha", "Hbeta"), None))
    assert tuple(d.name for d in out) == ("Halpha",)
