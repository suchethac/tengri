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
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.dust._params import PARAMS as _DUST_EMISSION_PARAMS
from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.components.sed_model_component import _REGISTRY
from tengri.forward.orchestrator import default_params_dict
from tengri.protocols.component import ForwardState, declared_default

pytestmark = pytest.mark.contract

#: Rest-frame grid: 100 A to 1 cm, wide enough for every registered
#: emission engine's IR peak (mirrors tests/integration/test_dust_emission_pipeline.py).
_WAVE = jnp.logspace(2.0, 8.0, 256)


def _registered_emission_components() -> dict[str, type]:
    """Every concrete :class:`EmissionComponent` subclass in ``_REGISTRY``."""
    return {
        name: cls
        for name, cls in _REGISTRY.items()
        if isinstance(cls, type) and issubclass(cls, EmissionComponent)
    }


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


@pytest.mark.parametrize("name", sorted(_registered_emission_components()))
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
