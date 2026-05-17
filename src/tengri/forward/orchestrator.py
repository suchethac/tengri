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
    DerivedKey,
    ParamDeclaration,
    PipelineContractError,
    PipelineState,
    SEDComponent,
)

__all__ = [
    "merge_declared_parameters",
    "run_components",
    "sample_params_dict",
    "slice_params_for_component",
    "validate_pipeline",
]


# Canonical units for every well-known cross-component derived key. Both
# publisher and consumer must declare a units string matching the table
# entry (where one exists). Adding a new derived key to the contract is
# a one-line edit here, in the same PR that introduces the publisher.
# Same friction model as ``tools/check_param_prefixes.py::ALLOWED_PREFIXES``.
_CANONICAL_UNITS: dict[str, str] = {
    # Stellar — surviving / formed mass + SFR variants
    "log_mstar": "dex",
    "log_mstar_formed": "dex",
    "sfr": "Msun/yr",
    "sfr_10myr": "Msun/yr",
    "sfr_100myr": "Msun/yr",
    # Stellar — age-resolved tensors (hard requires for nebular / dust2c)
    "L_age": "erg/s",
    "lnu_age": "erg/s/Hz",
    "ssp_ages_yr": "yr",
    "age_weights": "Msun",
    # Stellar — ionizing rate + SFH grid + chemistry history
    "nion": "photons/s",
    "sfh_grid_lbt_yr": "yr",
    "sfr_history": "Msun/yr",
    "log_metallicity_history": "dex",
    # Dust attenuation / emission outputs
    "L_ir": "erg/s",
    "L_absorbed": "erg/s",
    "dust_attenuation_factor": "",
    "sed_dust_attenuated": "erg/s/Hz",
    "sed_dust_ir": "erg/s/Hz",
    # AGN outputs
    "L_agn_bol": "erg/s",
    "L_agn_torus": "erg/s",
    "L_agn_absorbed": "erg/s",
    "sed_agn": "erg/s/Hz",
    "sed_grahsp": "erg/s/Hz",
    # Nebular outputs (continuous SED in erg/s/Hz per 2026-04-08 standard;
    # discrete line/continuum primitives are Lsun per the 2026-05-17
    # convention — see project_nebular_unit_conventions memory entry).
    "sed_nebular": "erg/s/Hz",
    "sed_shock": "erg/s/Hz",
    "line_waves": "Angstrom",
    "line_lums": "erg/s",
    # Radio / X-ray / IGM
    "sed_radio": "erg/s/Hz",
    "sed_xray": "erg/s/Hz",
    "igm_transmission": "",
    # Shock (MAPPINGS path)
    "shock_log_lhalpha": "dex",
}


# Pairs of components that legitimately publish the same key as alternate
# implementations of the same role. The validator allows at most one of
# each pair to appear in a pipeline at once (handled implicitly: the
# build_components factory selects one variant by config), but does not
# treat their shared key as a duplicate-publish error.
_ALTERNATE_PUBLISHERS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"DustSEDComponent", "DustAttenuationSEDComponent"}),
        frozenset({"AGNSEDComponent", "GRAHSPSEDComponent"}),
    }
)


def _levenshtein(a: str, b: str) -> int:
    """Plain Levenshtein distance — used only for "Did you mean: ..." hints."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(
                min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1))
            )
        prev = curr
    return prev[-1]


def _did_you_mean(missing: str, known: Iterable[str]) -> str:
    """Return ' (Did you mean: <closest>?)' when a known key is within edit distance 2."""
    best_name = None
    best_dist = 3  # strictly less than 3 to suggest
    for k in known:
        d = _levenshtein(missing, k)
        if d < best_dist:
            best_dist = d
            best_name = k
    if best_name is None:
        return ""
    return f" (Did you mean: {best_name!r}?)"


def validate_pipeline(components: Iterable[SEDComponent]) -> None:
    r"""Check the publish / require contract over an ordered component list.

    Runs at :class:`tengri.SEDModel` construction time, once per model.
    Zero hot-path / JIT cost — all checks are pure-Python over the
    metadata returned by :meth:`SEDComponent.publishes` and
    :meth:`SEDComponent.requires`.

    The five checks
    ---------------
    1. **Duplicate publish.** Two components publish the same key — unless
       they are listed in :data:`_ALTERNATE_PUBLISHERS` as alternate
       implementations of the same role (e.g. one-component vs two-component
       dust).
    2. **Missing publisher.** A key declared in :meth:`requires` has no
       upstream :meth:`publishes`. The error message includes a
       ``Did you mean: ...`` suggestion when a published key is within edit
       distance 2 — this is what catches the silent-rename hazard
       (``L_ir`` → ``L_dust_total`` produces a suggestion).
    3. **Out-of-order publisher.** A required key is published by a
       component that appears *after* the consumer in the pipeline list.
    4. **Unit mismatch (publisher vs consumer).** Consumer's
       ``DerivedKey.units`` differs from publisher's. The validator
       refuses to silently paper over a unit disagreement; conversion is
       the publisher's or consumer's responsibility, at their own
       boundary, made explicit.
    5. **Canonical-units mismatch.** Either side declares a units string
       that differs from :data:`_CANONICAL_UNITS` for that key (where the
       table has an entry). Catches the case where a *new* component
       independently invents a units convention — the contract pins the
       project-wide answer.

    Parameters
    ----------
    components : iterable of SEDComponent
        Ordered list of pipeline components, in the order they will run.

    Raises
    ------
    PipelineContractError
        On any of the five failure modes. The message names the offending
        component class, the offending key, the offending units strings,
        and (where applicable) a ``Did you mean: ...`` suggestion.
    """
    component_list = list(components)

    # Stage 1: build the publish map (component-order indexed) and check
    # duplicate publish + canonical-units agreement on the publish side.
    publishers: dict[str, tuple[int, SEDComponent, DerivedKey]] = {}
    for idx, component in enumerate(component_list):
        for key in component.publishes():
            if not isinstance(key, DerivedKey):
                raise PipelineContractError(
                    f"Component {type(component).__name__!r} publishes() returned "
                    f"a non-DerivedKey entry of type {type(key).__name__}"
                )
            canonical = _CANONICAL_UNITS.get(key.name)
            if canonical is not None and key.units != canonical:
                raise PipelineContractError(
                    f"Component {type(component).__name__!r} publishes "
                    f"{key.name!r} in {key.units!r} but the canonical-units "
                    f"table pins it to {canonical!r}. Either fix the units "
                    f"string or update _CANONICAL_UNITS in "
                    f"tengri.forward.orchestrator (in the same PR)."
                )
            if key.name in publishers:
                prior_idx, prior_comp, _ = publishers[key.name]
                pair = frozenset(
                    {type(prior_comp).__name__, type(component).__name__}
                )
                if pair not in _ALTERNATE_PUBLISHERS:
                    raise PipelineContractError(
                        f"Derived key {key.name!r} is published by both "
                        f"{type(prior_comp).__name__!r} (at position {prior_idx}) "
                        f"and {type(component).__name__!r} (at position {idx}). "
                        f"Two components cannot publish the same key unless "
                        f"they are registered as alternate implementations "
                        f"in _ALTERNATE_PUBLISHERS."
                    )
            publishers[key.name] = (idx, component, key)

    # Stage 2: walk consumers in order; check each required key is
    # published by an *earlier* component with matching units.
    for idx, component in enumerate(component_list):
        for needed in component.requires():
            if not isinstance(needed, DerivedKey):
                raise PipelineContractError(
                    f"Component {type(component).__name__!r} requires() returned "
                    f"a non-DerivedKey entry of type {type(needed).__name__}"
                )
            canonical = _CANONICAL_UNITS.get(needed.name)
            if canonical is not None and needed.units != canonical:
                raise PipelineContractError(
                    f"Component {type(component).__name__!r} requires "
                    f"{needed.name!r} in {needed.units!r} but the "
                    f"canonical-units table pins it to {canonical!r}."
                )
            if needed.name not in publishers:
                hint = _did_you_mean(needed.name, publishers.keys())
                raise PipelineContractError(
                    f"Component {type(component).__name__!r} requires derived "
                    f"key {needed.name!r} (in {needed.units!r}) but no "
                    f"upstream component publishes it.{hint}"
                )
            pub_idx, pub_comp, pub_key = publishers[needed.name]
            if pub_idx >= idx:
                raise PipelineContractError(
                    f"Component {type(component).__name__!r} (position {idx}) "
                    f"requires {needed.name!r} but it is published by "
                    f"{type(pub_comp).__name__!r} at position {pub_idx} — "
                    f"the publisher must come strictly before the consumer."
                )
            if pub_key.units != needed.units:
                raise PipelineContractError(
                    f"Component {type(component).__name__!r} requires "
                    f"{needed.name!r} in {needed.units!r} but "
                    f"{type(pub_comp).__name__!r} publishes it in "
                    f"{pub_key.units!r}. The contract refuses to silently "
                    f"convert; fix one of the units strings (the canonical "
                    f"answer is in _CANONICAL_UNITS)."
                )


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
