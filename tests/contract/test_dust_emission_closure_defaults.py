# SPDX-License-Identifier: BSD-3-Clause
"""Every analytic dust-emission closure's signature default matches its component's own.

#2241: ``_closures.py``'s pure emission functions (``modified_blackbody``,
``graybody``, ``casey2012``, ``schreiber2016``) and the
``SEDModelComponent`` subclasses that wrap them
(``ModifiedBlackbodyIRSEDComponent`` and siblings) each used to spell the
same physical default (``dust_T``, ``dust_beta_ir``, ``dust_lambda_0_um``,
``dust_alpha_mir``, ``dust_epsilon_mbb``, ``dust_f_pah``) as an independent
bare numeral. Two copies of one value drift silently: this census pins that
each closure's signature default and its paired component's own declared
default read the SAME value, for every parameter the two share.

This is read through the component's own ``declared_parameters()`` (i.e. its
class-level :class:`~tengri.parameters.priors.Distribution` attributes), not
through ``tengri.parameters.registry.registry()``: the shared
``components/dust/_params.py`` table disagrees with these four components'
own ``dust_T``/``dust_beta_ir`` defaults (#2261, filed separately -- this
census is not the place that resolves it), so comparing against the global
registry would be comparing against the wrong source of truth for exactly
the two names that already do not agree with it.

Mutation-check (recorded in the #2241 PR body): re-hardcoding one closure's
signature default to a WRONG numeral (e.g. ``modified_blackbody(...,
dust_T: float = 35.0)``, casey2012's temperature copy-pasted over
modified_blackbody's own 30.0) makes this test fail immediately (30.0 !=
35.0) and also trips ``tools/check_literal_param_defaults.py`` independently
-- the two checks catch the same drift by different mechanisms (value
comparison here; source-level literal detection there). A same-VALUE
re-hardcode (e.g. ``graybody(..., dust_lambda_0_um: float = 200.0)``, which
still equals the component's own 200.0) is caught by the guard alone: no
runtime value comparison can distinguish "read via a shared constant" from
"independently spelled out and coincidentally still correct".
"""

from __future__ import annotations

import inspect

import pytest

from tengri.components.dust.emission.analytic._closures import (
    casey2012,
    graybody,
    modified_blackbody,
    schreiber2016,
)
from tengri.components.dust.emission.analytic.casey2012 import Casey2012IRSEDComponent
from tengri.components.dust.emission.analytic.graybody import GraybodyIRSEDComponent
from tengri.components.dust.emission.analytic.modified_blackbody import (
    ModifiedBlackbodyIRSEDComponent,
)
from tengri.components.dust.emission.analytic.schreiber2016 import (
    Schreiber2016AnalyticIRSEDComponent,
)

pytestmark = pytest.mark.contract

#: (closure, paired SEDModelComponent class). Each component's own
#: ``declared_parameters()`` is the source of truth this test reads against
#: -- NOT ``tengri.parameters.registry.registry()`` (see module docstring).
_CLOSURE_COMPONENT_PAIRS = (
    (modified_blackbody, ModifiedBlackbodyIRSEDComponent),
    (graybody, GraybodyIRSEDComponent),
    (casey2012, Casey2012IRSEDComponent),
    (schreiber2016, Schreiber2016AnalyticIRSEDComponent),
)


def _component_declared_defaults(component_cls: type) -> dict[str, float]:
    """``{full_param_name: declared_default}`` for one component instance."""
    instance = component_cls()
    return {
        declaration.name: float(declaration.prior.default)
        for declaration in instance.declared_parameters()
        if getattr(declaration.prior, "default", None) is not None
    }


@pytest.mark.parametrize(
    "closure,component_cls",
    _CLOSURE_COMPONENT_PAIRS,
    ids=[closure.__name__ for closure, _ in _CLOSURE_COMPONENT_PAIRS],
)
def test_closure_default_matches_component_declaration(closure, component_cls):
    """Every parameter the closure and its component share must agree.

    Iterates the INTERSECTION of the closure's signature parameter names and
    the component's own declared names, so a parameter the component does
    not declare (``energy_balance_split``'s shared-table knobs have no
    per-component analog here) or the closure does not accept is simply
    not compared -- this is a positive assertion over the shared set, not a
    completeness census.
    """
    component_defaults = _component_declared_defaults(component_cls)
    signature = inspect.signature(closure)

    shared = sorted(set(signature.parameters) & set(component_defaults))
    assert shared, (
        f"{closure.__name__} and {component_cls.__name__} share no declared "
        "parameter names -- the pairing above may be wrong."
    )

    mismatches = []
    for name in shared:
        closure_default = signature.parameters[name].default
        component_default = component_defaults[name]
        if closure_default != component_default:
            mismatches.append((name, closure_default, component_default))

    assert not mismatches, (
        f"{closure.__name__} and {component_cls.__name__} disagree on: "
        + ", ".join(f"{name} (closure={c!r}, component={p!r})" for name, c, p in mismatches)
    )


def test_every_registered_analytic_component_is_covered():
    """The four analytic templates with their own class-level declarations
    are all paired above; a fifth one added later must extend this list."""
    covered = {cls.__name__ for _, cls in _CLOSURE_COMPONENT_PAIRS}
    expected = {
        "ModifiedBlackbodyIRSEDComponent",
        "GraybodyIRSEDComponent",
        "Casey2012IRSEDComponent",
        "Schreiber2016AnalyticIRSEDComponent",
    }
    assert covered == expected
