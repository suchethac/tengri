# SPDX-License-Identifier: BSD-3-Clause
"""Every registered dust-emission component's own declarations are enough to run it.

#2167 added ``dust_lambda_0_um`` to ``casey2012``/``graybody``'s class-level
parameter declarations. Nothing that runs in the fast tier noticed: eight
``test_full_chain_composability[casey2012-*]`` parametrizations in
``tests/integration/test_components_jit.py`` hand-listed ``dust_T`` /
``dust_beta_ir`` instead of reading the component's own declarations, and
started raising ``KeyError: 'lambda_0_um'`` -- but ``tests/integration`` is
auto-marked ``slow`` and only runs in the schedule-gated nightly, so the PR
that added the parameter was green and the tier went red the next morning
(#2236).

This census closes that gap in the fast tier. For every dust-emission type
registered in :data:`tengri.components.sed_model_component._REGISTRY`, it
builds a parameter dict from the component's *own* ``declared_parameters()``
(via :func:`tengri.forward.orchestrator.default_params_dict`, the same
mechanism ``test_components_jit.py`` now uses) and calls :meth:`apply` once
on a small synthetic grid -- deliberately bypassing ``SEDModel.build`` /
``Parameters``, whose grammar already supplies every declared default and
so would not have reproduced #2167's failure. A future parameter a
component's ``predict`` reads but its class-level declarations do not
supply (the #2167 shape) raises ``KeyError`` here; a declared parameter
``predict`` never reads at all raises nothing but is caught by the
companion input census, :mod:`tests.contract.test_component_declared_inputs_census`.

``energy_balance_split`` declares its knobs globally in
``components/dust/_params.py`` rather than on the class (see that module's
docstring and ``tests/regression/bug/test_energy_balance_split_wildcard_scope.py``),
so its own ``declared_parameters()`` is empty by design; its
``reads_parameters`` marker names the keys instead, and this test reads
their defaults from that shared table.
:func:`test_energy_balance_split_missing_shared_param_raises` exercises the
gap this component's own declarations cannot: since ``predict`` now
subscripts ``p["f_cold"]`` and siblings directly rather than falling back to
a stale literal (#2241), a ``reads_parameters`` name missing from the
supplied dict raises ``KeyError`` here exactly the way a #2167-shaped gap
does for every other registered type.

**Data gate.** ``draine2021_pah_ir`` needs the untracked, gitignored
``data/pahspec_draine2021.h5`` (104 MB, not on any CI runner). Unlike every
other component here, its ``predict``/``load`` never lets a missing-file
``FileNotFoundError`` escape: ``draine2021_pah_ir.py`` catches the failure
itself, warns, and returns ``(sed_in, {})`` -- the documented #1278
silent-emitter contract, pinned by
``test_bug_1278_draine2021_pah_silent_emitter.py``. Calling ``apply()`` on
this type without the grid therefore does not raise or skip; it returns an
unchanged SED with no ``sed_dust_ir`` key, which this test's own assertions
would (correctly) fail on. So this one case is gated on
``tests._data_skip.requires_pahspec`` *before* ``apply()`` runs, exactly as
``tests/contract/test_dust_emission_exact_energy_balance.py`` already gates
its PAH parametrization -- reusing the same ``PAHSPEC_EMISSION_TYPES`` /
``has_pahspec`` / ``requires_pahspec`` helpers rather than a second copy of
the gate. The ``except FileNotFoundError`` handler below is kept because
other template-backed types (``dale2014``, ``draine_li2007``,
``draine_li2014``, and siblings) load their tracked ``data/*.h5`` grid via a
lazy loader that *does* raise a plain ``FileNotFoundError`` if that file is
ever genuinely absent (checked: ``emission.py``'s ``_make_lazy_loader`` /
``_dl07_lazy_wrapper``) -- a broken-checkout signal for a file that is
supposed to always be there, not an optional bundle, so it is reported as a
per-case skip rather than a hard failure. A final census test asserts the
skipped set is exactly the PAH data gate, so a skip for any other reason
still fails the file.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.dust._params import PARAMS as _DUST_EMISSION_PARAMS
from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.components.sed_model_component import _REGISTRY
from tengri.forward.orchestrator import default_params_dict
from tengri.protocols.component import ForwardState, declared_default
from tests._data_skip import PAHSPEC_EMISSION_TYPES, has_pahspec, requires_pahspec

pytestmark = pytest.mark.contract

#: Rest-frame grid: 100 A to 1 cm, wide enough for every registered
#: emission engine's IR peak (mirrors tests/integration/test_dust_emission_pipeline.py).
_WAVE = jnp.logspace(2.0, 8.0, 256)

#: Names this run actually completed (appended at the end of the test body,
#: so a skip -- marker-based or the runtime FileNotFoundError fallback --
#: never adds its name here). Read by the final census test.
_COMPLETED: list[str] = []


def _registered_emission_components() -> dict[str, type]:
    """Every concrete :class:`EmissionComponent` subclass in ``_REGISTRY``."""
    return {
        name: cls
        for name, cls in _REGISTRY.items()
        if isinstance(cls, type) and issubclass(cls, EmissionComponent)
    }


def _emission_component_params() -> list:
    """Parametrize cases for every registered type, PAH gated on its grid.

    Mirrors ``tests/contract/test_dust_emission_exact_energy_balance.py``'s
    ``_all_balanced_type_params``: a name in ``PAHSPEC_EMISSION_TYPES`` gets
    the ``requires_pahspec`` marker, so it skips before the test body (and
    therefore ``apply()``) ever runs.
    """
    return [
        pytest.param(name, marks=requires_pahspec) if name in PAHSPEC_EMISSION_TYPES else name
        for name in sorted(_registered_emission_components())
    ]


def _params_for(component: EmissionComponent) -> dict[str, jnp.ndarray]:
    """Build a params dict from ``component``'s own declarations (+ ``redshift``).

    Supplements with :attr:`reads_parameters` defaults (read from the shared
    ``components/dust/_params.py`` table) for a component whose knobs are
    declared elsewhere by design -- not a #2167-shaped gap.
    """
    own = default_params_dict([component], overrides={"redshift": 0.0})
    shared_names = getattr(component, "reads_parameters", frozenset()) - set(own)
    for name in shared_names:
        own[name] = jnp.asarray(declared_default(_DUST_EMISSION_PARAMS, name))
    return own


@pytest.mark.parametrize("name", _emission_component_params())
def test_declared_defaults_are_sufficient_for_apply(name):
    """``default_params_dict([component])`` alone must be enough for ``apply()``.

    Regression guard for #2167 / #2236: a hand-built params dict that lists a
    dust-emission component's parameters by name, rather than reading them off
    the component's own declarations, silently strands the moment a new
    parameter (like casey2012's ``dust_lambda_0_um``) is declared. Building
    the dict this way instead means a stranding parameter fails here -- in the
    fast ``contract`` tier -- rather than only where a hand-built fixture in
    the schedule-gated nightly happens to hit it.
    """
    component = _registered_emission_components()[name]()
    state = ForwardState(
        wave=_WAVE,
        sed_intrinsic=jnp.zeros_like(_WAVE),
        derived={"L_ir": 1e44, "log_L_ir": jnp.log10(1e44)},
    )
    params = _params_for(component)

    try:
        out = component.apply(state, params)
    except FileNotFoundError as exc:
        pytest.skip(f"{name}: required template/grid data not present on this machine ({exc})")

    assert out.sed_intrinsic is not None
    assert jnp.all(jnp.isfinite(out.sed_intrinsic)), (
        f"{name}: apply() using only its own declared defaults produced non-finite output."
    )
    assert "sed_dust_ir" in out.derived
    _COMPLETED.append(name)


def test_energy_balance_split_missing_shared_param_raises():
    """A ``reads_parameters`` supply gap on ``energy_balance_split`` now
    raises, mirroring how a #2167-shaped gap fails every other type above.

    Deliberately does NOT add to ``_COMPLETED``: this is a negative case
    (asserts ``apply()`` raises), not a member of the sufficiency
    parametrization above.

    Before #2241, ``EnergyBalanceSplitIRSEDComponent.predict`` read this
    knob via ``p.get("f_cold", 0.5)``, so deleting it from the supplied
    dict below would have silently substituted the stale literal and
    returned a finite SED instead of raising -- exactly the vacuous pass
    the module docstring used to flag as a known limit of this census.
    """
    component = _registered_emission_components()["energy_balance_split"]()
    state = ForwardState(
        wave=_WAVE,
        sed_intrinsic=jnp.zeros_like(_WAVE),
        derived={"L_ir": 1e44, "log_L_ir": jnp.log10(1e44)},
    )
    params = _params_for(component)
    del params["dust_f_cold"]  # apply() strips the "dust_" prefix before predict()

    with pytest.raises(KeyError, match="f_cold"):
        component.apply(state, params)


def test_census_only_the_pahspec_gate_may_skip():
    """A skip for any reason other than the PAH data gate fails here.

    Runs last in file order (mirrors
    ``test_dust_emission_exact_energy_balance.py::test_census_every_required_type_completed``):
    the only skip this census accepts is ``PAHSPEC_EMISSION_TYPES`` when the
    untracked grid is genuinely absent. Anything else missing from
    ``_COMPLETED`` -- a marker misfire, or the runtime ``FileNotFoundError``
    fallback catching a genuinely broken checkout for a tracked grid -- fails
    the file instead of leaving a silent, unaccounted skip.
    """
    all_names = set(_registered_emission_components())
    data_gated = set() if has_pahspec() else (all_names & PAHSPEC_EMISSION_TYPES)
    missing = sorted(all_names - set(_COMPLETED) - data_gated)
    assert not missing, (
        f"dust-emission types did not complete the declared-params census: {missing}"
    )
