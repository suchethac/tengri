# SPDX-License-Identifier: BSD-3-Clause
"""Minimal pipeline runner over a list of :class:`SEDComponent` adapters.

This is the smallest possible consumer of the Phase II-1 ``core/``
Protocol scaffold. It exists to drive the contract tests for the first
two adapters (RadioSEDComponent, IGMSEDComponent) — **not** to replace
:class:`tengri.SEDModel`'s tier-dispatch path.

The two-adapter rule
--------------------
Per ``~/.claude/skills/improve-codebase-architecture/LANGUAGE.md``:

    *One adapter = hypothetical seam. Two adapters = real seam.*

Until two real components run through this orchestrator, the
:class:`tengri.core.SEDComponent` Protocol is hypothetical. This file
makes the seam real.

What this is **not**
--------------------
- Not the migration of :class:`tengri.SEDModel`. The legacy code path
  is untouched.
- Not a public API. Lives in ``forward/`` as an internal Phase II-1
  artefact; will be promoted (or replaced) when Phase II-2 starts.
- Not a registry. Components are passed in as a plain ordered list.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax.numpy as jnp

from tengri.core.component import (
    BARE_NAME_ALLOWLIST,
    ParamDeclaration,
    PipelineState,
    SEDComponent,
)

__all__ = [
    "merge_declared_parameters",
    "run_components",
    "sample_params_dict",
    "slice_params_for_component",
]


def slice_params_for_component(
    component: SEDComponent,
    params: Mapping[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    r"""Return params visible to ``component``.

    A component sees:

    1. Every key starting with its :attr:`SEDComponent.parameter_prefix`.
    2. Every key in :data:`BARE_NAME_ALLOWLIST` (currently ``redshift``).

    Notes
    -----
    A component with ``parameter_prefix == ""`` would see *everything*
    via rule 1 — that's why ``""`` is forbidden by the contract test.
    Components that need bare names must declare a non-empty prefix and
    rely on the allowlist for shared scalars.
    """
    prefix = component.parameter_prefix
    # Phase II-2: prefix may be a single str OR a tuple of strings (e.g.
    # StellarSEDComponent owns ("sfh_", "met_", "chem_")). Normalise to
    # tuple here so downstream logic handles one shape only.
    prefixes = (prefix,) if isinstance(prefix, str) else tuple(prefix)
    if "" in prefixes or len(prefixes) == 0:
        raise ValueError(
            f"Component {component.name!r} declares an empty parameter_prefix. "
            f"Use the BARE_NAME_ALLOWLIST for shared scalars (e.g. redshift) "
            f"and give the component at least one non-empty prefix."
        )

    sliced = {k: v for k, v in params.items() if any(k.startswith(p) for p in prefixes)}
    for bare in BARE_NAME_ALLOWLIST:
        if bare in params:
            sliced[bare] = params[bare]
    return sliced


def merge_declared_parameters(
    components: Iterable[SEDComponent],
) -> dict[str, Any]:
    r"""Flatten ``declared_parameters()`` from each component into one prior dict.

    Each :class:`SEDComponent` owns a list of :class:`ParamDeclaration`
    entries (name + prior + description). This helper validates that:

    1. Every declared name starts with the owning component's
       :attr:`SEDComponent.parameter_prefix` *or* is in
       :data:`BARE_NAME_ALLOWLIST`.
    2. No two components declare the same parameter name (a collision
       would make the orchestrator's prefix-slicing ambiguous).

    The output maps each parameter name to its prior — suitable for
    spreading into :class:`tengri.Parameters` once the migration of
    :mod:`tengri.parameters.parameters` to a component-driven builder
    lands (Phase II-6 of the plan).

    Parameters
    ----------
    components : iterable of SEDComponent
        Adapters whose declarations should be merged.

    Returns
    -------
    dict mapping str -> Distribution
        Parameter name -> prior. Order is preserved by component order
        and within-component declaration order.

    Raises
    ------
    ValueError
        If a name violates the prefix rule or two components declare the
        same name.
    """
    out: dict[str, Any] = {}
    owners: dict[str, str] = {}  # name -> first owner (for collision msg)

    for component in components:
        prefix = component.parameter_prefix
        prefixes = (prefix,) if isinstance(prefix, str) else tuple(prefix)
        for decl in component.declared_parameters():
            if not isinstance(decl, ParamDeclaration):
                raise TypeError(
                    f"Component {component.name!r} declared a non-ParamDeclaration "
                    f"entry of type {type(decl).__name__}"
                )
            if (
                not any(decl.name.startswith(p) for p in prefixes)
                and decl.name not in BARE_NAME_ALLOWLIST
            ):
                raise ValueError(
                    f"Component {component.name!r} declared parameter "
                    f"{decl.name!r} which violates the prefix rule "
                    f"(expected one of prefixes {prefixes!r} or a name in "
                    f"BARE_NAME_ALLOWLIST {tuple(BARE_NAME_ALLOWLIST)})."
                )
            if decl.name in out:
                raise ValueError(
                    f"Parameter {decl.name!r} is declared by both "
                    f"{owners[decl.name]!r} and {component.name!r} — "
                    f"two components cannot own the same parameter."
                )
            out[decl.name] = decl.prior
            owners[decl.name] = component.name

    return out


def sample_params_dict(
    components: Iterable[SEDComponent],
    key: jnp.ndarray,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, jnp.ndarray]:
    r"""Sample one params dict from each component's declared priors.

    Closes the loop ``[components] → declarations → params dict →
    run_components``: a user can compose adapters, sample a draw, and
    immediately run the pipeline without manually typing a parameter
    dict.

    Parameters
    ----------
    components : iterable of SEDComponent
        Adapters whose declared priors should be sampled.
    key : jax.Array
        PRNG key. Split internally with :func:`jax.random.split` once
        per declared parameter.
    overrides : mapping, optional
        Fixed values that override the prior draw. Useful for pinning
        ``redshift`` or other scalars during a forward pass without
        building a custom prior.

    Returns
    -------
    dict mapping str -> jnp.ndarray
        Parameter name -> drawn value, ready to feed
        :func:`run_components`.

    Notes
    -----
    Not for inference — the priors here come from registry defaults,
    not user-tuned posteriors. Use :class:`tengri.Parameters` for
    proper inference workflows. This helper is for prior-predictive
    smoke tests and notebook demos.
    """
    import jax

    merged = merge_declared_parameters(components)
    overrides = overrides or {}
    keys = jax.random.split(key, max(len(merged), 1))
    out: dict[str, jnp.ndarray] = {}
    for (name, prior), subkey in zip(merged.items(), keys, strict=False):
        if name in overrides:
            out[name] = jnp.asarray(overrides[name])
        else:
            out[name] = jnp.asarray(prior.sample(subkey))
    # Bare-name allowlist entries that no component declared but the
    # caller still wants threaded through (typically redshift).
    for bare in BARE_NAME_ALLOWLIST:
        if bare in overrides and bare not in out:
            out[bare] = jnp.asarray(overrides[bare])
    return out


def run_components(
    components: Iterable[SEDComponent],
    state: PipelineState,
    params: Mapping[str, jnp.ndarray],
) -> PipelineState:
    r"""Thread ``state`` through ``components`` in order.

    Parameters
    ----------
    components : iterable of SEDComponent
        Ordered list. Each component reads what it needs from
        ``state`` and returns a *new* :class:`PipelineState`.
    state : PipelineState
        Initial pipeline state. Typically carries just ``wave`` and
        any seed values (e.g. ``sed_observed`` already populated).
    params : mapping
        Full parameter dict. Each component sees only its
        prefix-matched slice plus the bare-name allowlist.

    Returns
    -------
    PipelineState
        Final state after all components have applied.
    """
    for component in components:
        sliced = slice_params_for_component(component, params)
        state = component.apply(state, sliced)
    return state
