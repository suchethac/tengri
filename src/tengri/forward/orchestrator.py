# SPDX-License-Identifier: BSD-3-Clause
"""Run a list of :class:`SEDComponent` physics blocks in order.

Internal driver behind :class:`tengri.SEDModel`. Given an ordered list
of components (stellar, dust, nebular, AGN, IGM, radio, X-ray, …), it:

1. checks every component's declared inputs are produced upstream
   (:func:`validate_pipeline`),
2. threads a :class:`ForwardState` through each component's
   :meth:`SEDComponent.apply` in order, and
3. returns the resulting :class:`ForwardState` whose
   :attr:`~ForwardState.derived` carries the cross-component
   physics quantities (``L_ir``, ``lnu_age``, ``sed_nebular``, …).

Not a public API — astronomers use :class:`tengri.SEDModel` and never
touch this module directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import jax.numpy as jnp

from tengri.protocols.component import (
    BARE_NAME_ALLOWLIST,
    ComponentIOError,
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponent,
)

__all__ = [
    "merge_declared_parameters",
    "run_components",
    "sample_params_dict",
    "slice_params_for_component",
    "topological_sort",
    "validate_pipeline",
]


# ── Internal accessors for the output/input contract methods ───────
#
# outputs / inputs / optional_inputs are intentionally NOT part of the
# runtime-checkable SEDComponent Protocol surface (see ADR-0009,
# specifically the "drop from runtime-checkable Protocol" amendment).
# Components that have not been annotated contribute the safe empty-tuple
# default. These three helpers live at module scope so both
# validate_pipeline and topological_sort can reuse them.
#
# Renamed from ``outputs`` / ``inputs`` / ``optional_inputs``. For one
# minor version the accessors fall back to the old names with a
# ``DeprecationWarning`` so components can be migrated incrementally;
# the old names are removed in v1.0.


def _contract_method(c: SEDComponent, new: str, old: str) -> tuple[DerivedKey, ...]:
    fn = getattr(c, new, None)
    if callable(fn):
        return tuple(fn())
    legacy = getattr(c, old, None)
    if callable(legacy):
        import warnings

        warnings.warn(
            f"Component {type(c).__name__!r} declares `{old}()`; this is "
            f"deprecated and will be removed in tengri v1.0. Rename the "
            f"method to `{new}()`.",
            DeprecationWarning,
            stacklevel=3,
        )
        return tuple(legacy())
    return ()


def _outputs(c: SEDComponent) -> tuple[DerivedKey, ...]:
    return _contract_method(c, "outputs", "publishes")


def _inputs(c: SEDComponent) -> tuple[DerivedKey, ...]:
    return _contract_method(c, "inputs", "requires")


def _optional_inputs(c: SEDComponent) -> tuple[DerivedKey, ...]:
    return _contract_method(c, "optional_inputs", "requires_optional")


def components_consuming(
    component_list: Sequence[SEDComponent], key_name: str
) -> tuple[SEDComponent, ...]:
    """Components in a chain that declare ``key_name`` as an input.

    Reads the same ``inputs()`` / ``optional_inputs()`` contract the pipeline
    already validates (ADR-0009), so the answer is *derived* from the chain
    rather than restated in a second list.

    Parameters
    ----------
    component_list : sequence of SEDComponent
        The assembled component chain.
    key_name : str
        Derived-state key to look for, e.g. ``'sed_nebular'``.

    Returns
    -------
    tuple of SEDComponent
        Every component declaring ``key_name``, required or optional, in chain
        order. Empty when nothing consumes it.

    Notes
    -----
    **JIT-compatible**: not applicable — composition-time only.

    Exists so a fast path can ask "does anything still need this?" instead of
    asserting it from a hand-written census. ``sed_nebular`` was zeroed under
    the per-Q_H nebular grid on the stated grounds that its "only live
    consumers are the exact spectrum / dust-continuum paths"; the dust energy
    balance consumes it too, and a model with dust emission then re-emitted
    the stellar absorbed budget alone — 11 % low in the far-IR, with the
    posterior gradient up to 380 % wrong, silently and in float64. A census
    written next to the code it guards goes stale the first time a consumer is
    added somewhere else; a derived one cannot.
    """
    return tuple(
        c
        for c in component_list
        if any(k.name == key_name for k in (*_inputs(c), *_optional_inputs(c)))
    )


def _materialized(c: SEDComponent) -> SEDComponent:
    """A component's publishing variant, or the component itself.

    Duck-typed like the contract accessors above: a component with no
    publication shortcut needs no method and is returned by identity.
    """
    hook = getattr(c, "materialized", None)
    return hook() if callable(hook) else c


def materialized_chain(component_list: Sequence[SEDComponent]) -> tuple[SEDComponent, ...]:
    """Chain variant in which every component publishes the outputs it declares.

    A component may skip publishing an output when :func:`components_consuming`
    says nothing needs it — :class:`~tengri.components.nebular.component.NebularSEDComponent`
    zeroes ``sed_nebular`` under the per-Q_H grid, because skipping the Cue
    forward *is* the saving.

    That census reads the ADR-0009 contract, so it finds every consumer that
    declares an input and nothing else. A reader that takes a published key off
    ``state.derived`` without declaring one is invisible to it, and the forward
    state has several: ``state_to_sed_components`` behind
    ``Posterior.sed_components``, and the accumulated ``state.sed_intrinsic``
    behind ``pred.rest_sed()``. On a dust-free Cue model that cost the whole
    nebular continuum — ``sed_nebular`` exactly zero and the published SED 97 %
    short at its peak, in float64 and silently (#1673).

    Those callers ask for this chain rather than each teaching the census about
    itself. A component opts in by defining ``materialized()``; everything else
    is returned by identity.

    Parameters
    ----------
    component_list : sequence of SEDComponent
        The assembled component chain.

    Returns
    -------
    tuple of SEDComponent
        The chain in order, each publication-skipping component replaced by its
        publishing variant.

    Notes
    -----
    **JIT-compatible**: not applicable — composition-time only. Variants come
    from :func:`dataclasses.replace` over static config, so no JAX work happens
    here and the result is safe to build outside a trace and reuse inside one.
    """
    return tuple(_materialized(c) for c in component_list)


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
    # Stellar — age-resolved tensors (hard needs for nebular / dust2c)
    "L_age": "erg/s",
    "lnu_age": "erg/s/Hz",
    "ssp_ages_yr": "yr",
    "age_weights": "Msun",
    "log_stellar_mass_scale": "dex",
    # Stellar — ionizing rate + SFH grid + chemistry history
    "nion": "photons/s",
    "log_nion": "dex",
    "sfh_grid_lbt_yr": "yr",
    "sfr_history": "Msun/yr",
    "log_metallicity_history": "dex",
    # Stellar — photometry LUT (only published when
    # ``approx=WavePrecomp()`` is set on SEDModel).
    "stellar_phot_lnu_precomp": "erg/s/Hz",
    # Stellar — Taylor moment. Same conditions as
    # stellar_phot_lnu_precomp. Used by the filter-level dust integration.
    "stellar_phot_moment_precomp": "erg*Angstrom/s/Hz",
    # Stellar — age-resolved per-filter LUT (shape
    # (n_age, n_filter)). Sum over age axis equals stellar_phot_lnu_precomp.
    "stellar_phot_lnu_per_age_precomp": "erg/s/Hz",
    "stellar_phot_moment_per_age_precomp": "erg*Angstrom/s/Hz",
    # Filter pivot wavelengths (published by stellar when wave_precomp
    # is on; used by dust LUT, AGN LUT, IGM LUT downstream).
    "filter_eff_waves": "Angstrom",
    # Dust attenuation per filter. A(λ_eff) and A'(λ_eff).
    "dust_attenuation_precomp": "",
    "dust_attenuation_slope_precomp": "1/Angstrom",
    # Two-component dust. BC + diffuse layer precompute.
    "dust_bc_attenuation_precomp": "",
    "dust_bc_attenuation_slope_precomp": "1/Angstrom",
    "dust_diff_attenuation_precomp": "",
    "dust_diff_attenuation_slope_precomp": "1/Angstrom",
    "dust_young_indicator": "",
    # Dust attenuation / emission outputs
    "L_ir": "erg/s",
    "L_absorbed": "erg/s",
    "log_L_ir": "dex",
    "log_L_agn_bol": "dex",
    "dust_attenuation_factor": "",
    "sed_dust_attenuated": "erg/s/Hz",
    "sed_dust_ir": "erg/s/Hz",
    "L_ir_emission": "erg/s",
    # AGN outputs
    "L_agn_bol": "erg/s",
    "L_agn_torus": "erg/s",
    "L_agn_absorbed": "erg/s",
    "L_2500_intrinsic": "erg/s/Hz",
    "L_4400_intrinsic": "erg/s/Hz",
    "sed_agn": "erg/s/Hz",
    "sed_grahsp": "erg/s/Hz",
    # AGN — filter LUT (WavePrecomp).
    "agn_phot_lnu_precomp": "erg/s/Hz",
    # Nebular outputs (continuous SED in erg/s/Hz per the 2026-04-08
    # standard; discrete line/continuum primitives are Lsun per the
    # 2026-05-17 convention).
    "sed_nebular": "erg/s/Hz",
    "sed_shock": "erg/s/Hz",
    "line_waves": "Angstrom",
    "line_lums": "erg/s",
    # Nebular — photometry LUT (only non-BakedIn backends
    # publish, when ``approx=WavePrecomp()`` is set).
    "nebular_phot_lnu_precomp": "erg/s/Hz",
    # The same bucket with the young-limit screen integrated THROUGH each band,
    # published by the dust component from the reddened continuum (#1738). Replaces
    # the lambda_eff screening of the key above rather than adding to it — the
    # ``_attenuated_`` infix is what keeps it out of the ``*_phot_lnu_precomp``
    # summation sweep in ``predict_via_precomp``.
    "nebular_phot_lnu_attenuated_precomp": "erg/s/Hz",
    "nebular_restband_lnu_attenuated_precomp": "erg/s/Hz",
    # Shock (MAPPINGS V) — filter LUT. A separate additive component from the
    # photoionized nebular backend (#851), so it carries its own key (#1375).
    "shock_phot_lnu_precomp": "erg/s/Hz",
    # Spectrum LUT (published when approx=SpectrumPrecomp() is set).
    # Per-pixel rest-frame Lν at spectrum pixel centers.
    "spec_eff_waves": "Angstrom",
    "stellar_spec_lnu_precomp": "erg/s/Hz",
    "nebular_spec_lnu_precomp": "erg/s/Hz",
    "dust_spec_lnu_precomp": "erg/s/Hz",
    "agn_spec_lnu_precomp": "erg/s/Hz",
    "igm_spec_transmission_precomp": "",
    # Radio / X-ray / IGM
    "sed_radio": "erg/s/Hz",
    "sed_xray": "erg/s/Hz",
    "igm_transmission": "",
    # Shock (MAPPINGS path)
    "shock_log_lhalpha": "dex",
    # Spatial — 2D surface-brightness profile + the (x, y) kpc grid
    # underlying it. Published by spatial components (Sersic, Exponential,
    # FlatSlab, …). See architecture spec §3.3.
    "spatial_profile_2d": "",
    "spatial_grid_xy_kpc": "",
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


def _did_you_mean(missing: str, known: Iterable[str]) -> str:
    """Return ' (Did you mean: <closest>?)' when a known key is within edit distance 2."""
    from tengri.utils.strings import closest

    best_name = closest(missing, known, max_distance=2)
    return f" (Did you mean: {best_name!r}?)" if best_name else ""


def validate_pipeline(components: Iterable[SEDComponent]) -> None:
    r"""Check that every component's inputs are produced by upstream components.

    Runs at :class:`tengri.SEDModel` construction time, once per model.
    Zero hot-path / JIT cost — all checks are pure-Python over the
    metadata returned by :meth:`SEDComponent.outputs` and
    :meth:`SEDComponent.inputs`.

    The five checks
    ---------------
    1. **Duplicate output.** Two components publish the same key — unless
       they are listed in :data:`_ALTERNATE_PUBLISHERS` as alternate
       implementations of the same role (e.g. one-component vs two-component
       dust).
    2. **Missing producer.** A key declared in :meth:`inputs` has no
       upstream :meth:`outputs`. The error message includes a
       ``Did you mean: ...`` suggestion when a published key is within edit
       distance 2 — this is what catches the silent-rename hazard
       (``L_ir`` → ``L_dust_total`` produces a suggestion).
    3. **Out-of-order producer.** A required key is published by a
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
    ComponentIOError
        On any of the five failure modes. The message names the offending
        component class, the offending key, the offending units strings,
        and (where applicable) a ``Did you mean: ...`` suggestion.
    """
    component_list = list(components)

    # Stage 1: build the publish map (component-order indexed) and check
    # duplicate output + canonical-units agreement on the publish side.
    publishers: dict[str, tuple[int, SEDComponent, DerivedKey]] = {}
    for idx, component in enumerate(component_list):
        for key in _outputs(component):
            if not isinstance(key, DerivedKey):
                raise ComponentIOError(
                    f"Component {type(component).__name__!r} outputs() returned "
                    f"a non-DerivedKey entry of type {type(key).__name__}"
                )
            canonical = _CANONICAL_UNITS.get(key.name)
            if canonical is not None and key.units != canonical:
                raise ComponentIOError(
                    f"Component {type(component).__name__!r} outputs "
                    f"{key.name!r} in {key.units!r} but the canonical-units "
                    f"table pins it to {canonical!r}. Either fix the units "
                    f"string or update _CANONICAL_UNITS in "
                    f"tengri.forward.orchestrator (in the same PR)."
                )
            if key.name in publishers:
                prior_idx, prior_comp, _ = publishers[key.name]
                pair = frozenset({type(prior_comp).__name__, type(component).__name__})
                if pair not in _ALTERNATE_PUBLISHERS:
                    raise ComponentIOError(
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
        for needed in _inputs(component):
            if not isinstance(needed, DerivedKey):
                raise ComponentIOError(
                    f"Component {type(component).__name__!r} inputs() returned "
                    f"a non-DerivedKey entry of type {type(needed).__name__}"
                )
            canonical = _CANONICAL_UNITS.get(needed.name)
            if canonical is not None and needed.units != canonical:
                raise ComponentIOError(
                    f"Component {type(component).__name__!r} needs "
                    f"{needed.name!r} in {needed.units!r} but the "
                    f"canonical-units table pins it to {canonical!r}."
                )
            if needed.name not in publishers:
                hint = _did_you_mean(needed.name, publishers.keys())
                raise ComponentIOError(
                    f"Component {type(component).__name__!r} needs derived "
                    f"key {needed.name!r} (in {needed.units!r}) but no "
                    f"upstream component outputs it.{hint}"
                )
            pub_idx, pub_comp, pub_key = publishers[needed.name]
            if pub_idx >= idx:
                raise ComponentIOError(
                    f"Component {type(component).__name__!r} (position {idx}) "
                    f"needs {needed.name!r} but it is published by "
                    f"{type(pub_comp).__name__!r} at position {pub_idx} — "
                    f"the producer must come strictly before the consumer."
                )
            if pub_key.units != needed.units:
                raise ComponentIOError(
                    f"Component {type(component).__name__!r} needs "
                    f"{needed.name!r} in {needed.units!r} but "
                    f"{type(pub_comp).__name__!r} outputs it in "
                    f"{pub_key.units!r}. The contract refuses to silently "
                    f"convert; fix one of the units strings (the canonical "
                    f"answer is in _CANONICAL_UNITS)."
                )

    # Stage 3: optional reads — Phase B of issue #21. Same checks as
    # required reads EXCEPT that a missing publisher is OK (the
    # consumer has a documented fallback). Catches a future publisher
    # rename or unit drift without forcing every pipeline to instantiate
    # the optional upstream component.
    for idx, component in enumerate(component_list):
        for needed in _optional_inputs(component):
            if not isinstance(needed, DerivedKey):
                raise ComponentIOError(
                    f"Component {type(component).__name__!r} optional_inputs() "
                    f"returned a non-DerivedKey entry of type {type(needed).__name__}"
                )
            canonical = _CANONICAL_UNITS.get(needed.name)
            if canonical is not None and needed.units != canonical:
                raise ComponentIOError(
                    f"Component {type(component).__name__!r} optionally needs "
                    f"{needed.name!r} in {needed.units!r} but the "
                    f"canonical-units table pins it to {canonical!r}."
                )
            if needed.name not in publishers:
                continue  # no upstream publisher → fallback applies; not an error
            pub_idx, pub_comp, pub_key = publishers[needed.name]
            if pub_idx >= idx:
                raise ComponentIOError(
                    f"Component {type(component).__name__!r} (position {idx}) "
                    f"optionally needs {needed.name!r} but it is published by "
                    f"{type(pub_comp).__name__!r} at position {pub_idx} — "
                    f"the producer must come strictly before the consumer."
                )
            if pub_key.units != needed.units:
                raise ComponentIOError(
                    f"Component {type(component).__name__!r} optionally needs "
                    f"{needed.name!r} in {needed.units!r} but "
                    f"{type(pub_comp).__name__!r} outputs it in "
                    f"{pub_key.units!r}. The contract refuses to silently "
                    f"convert; fix one of the units strings (the canonical "
                    f"answer is in _CANONICAL_UNITS)."
                )


def topological_sort(components: Iterable[SEDComponent]) -> list[SEDComponent]:
    r"""Stable topological sort over the output/input dependency graph.

    Produces an ordering where every component appears strictly after
    every other component whose :meth:`outputs` it consumes via
    :meth:`inputs` or :meth:`optional_inputs`. Among components with
    no ordering constraint, the input order is preserved (stable sort).

    This is the inverse of :func:`validate_pipeline`'s "out-of-order
    publisher" check — instead of refusing pipelines whose hand-coded
    order violates declared dependencies, it *derives* the order from
    the declarations. See ADR-0006.

    Parameters
    ----------
    components : iterable of SEDComponent
        The unordered (or arbitrarily-ordered) component list. Typically
        passed straight from :func:`tengri.forward.build_components`
        which appends in domain-grouped order; the sort tightens that
        into the dependency-respecting order before downstream consumers
        observe it.

    Returns
    -------
    list of SEDComponent
        Topologically ordered. For the canonical pipeline (stellar,
        nebular, AGN, dust, radio, X-ray, IGM), this reproduces the
        hand-coded order byte-for-byte — the snapshot test in
        :mod:`tests.integration.test_derived_contract_snapshots` is the
        regression guarantee.

    Raises
    ------
    ComponentIOError
        If the dependency graph contains a cycle. The error message
        names every component still pending when the algorithm stalls
        — typically the cycle's participants.

    Notes
    -----
    **Algorithm.** Kahn's algorithm with stable tie-breaking: at each
    step, the lowest-input-index component whose dependencies are all
    already emitted is picked next. With deterministic-order producer
    resolution (first publisher wins on duplicates — see
    :data:`_ALTERNATE_PUBLISHERS`), the sort is fully deterministic.

    **Why both ``inputs`` and ``optional_inputs``.** A hard
    requirement establishes ordering by definition. An optional
    requirement *also* establishes ordering when the publisher is
    present: the consumer reads from ``state.derived`` with a fallback,
    so it can only read meaningful data if the publisher has already
    written. The validator enforces strict-before for both flavors
    (ADR-0004 Phase B); the sort must too, else
    :func:`validate_pipeline` would reject sort output.

    **Zero JIT cost.** Runs once at :func:`build_components` time.
    """
    component_list = list(components)
    n = len(component_list)

    # Map each published derived-key name to the index of its publisher.
    # First-publisher-wins matches the validator's
    # ``_ALTERNATE_PUBLISHERS`` resolution: when two components publish
    # the same key as alternate implementations of the same role, the
    # first one in the input list takes precedence.
    producer: dict[str, int] = {}
    for i, c in enumerate(component_list):
        for key in _outputs(c):
            producer.setdefault(key.name, i)

    # Build incoming edges per node. An edge i ← j means "i needs
    # something j outputs, so j must come before i". Both hard and
    # optional needs contribute.
    deps: list[set[int]] = [set() for _ in range(n)]
    for i, c in enumerate(component_list):
        for needed in (*_inputs(c), *_optional_inputs(c)):
            pub = producer.get(needed.name)
            if pub is not None and pub != i:
                deps[i].add(pub)

    # Kahn's algorithm with stable tie-break.
    emitted: list[int] = []
    emitted_set: set[int] = set()
    remaining = set(range(n))
    while remaining:
        # Lowest input-order index whose dependencies are all already
        # emitted. Linear scan; n is tiny (typically 7) so this is fine.
        picked: int | None = None
        for i in range(n):
            if i in remaining and deps[i].issubset(emitted_set):
                picked = i
                break
        if picked is None:
            pending_names = [type(component_list[i]).__name__ for i in sorted(remaining)]
            raise ComponentIOError(
                f"Topological sort failed: cycle in output/input graph "
                f"involving {pending_names!r}. Every inputs() / "
                f"optional_inputs() declaration must be satisfiable in some "
                f"linear order — check the offending components."
            )
        emitted.append(picked)
        emitted_set.add(picked)
        remaining.remove(picked)

    return [component_list[i] for i in emitted]


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
    # ``parameter_prefix`` may be a single str or a tuple of strings (e.g.
    # StellarSEDComponent owns ("sfh_", "met_", "chem_")). Normalize to
    # tuple so downstream logic handles one shape only.
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
    spreading into :class:`tengri.Parameters` once the component-driven
    builder in :mod:`tengri.parameters.parameters` lands.

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
    state: ForwardState,
    params: Mapping[str, jnp.ndarray],
    ssp_data: Any | None = None,
    template_data: Any | None = None,
) -> ForwardState:
    r"""Thread ``state`` through ``components`` in order.

    Parameters
    ----------
    components : iterable of SEDComponent
        Ordered list. Each component reads what it needs from
        ``state`` and returns a *new* :class:`ForwardState`.
    state : ForwardState
        Initial pipeline state. Typically carries just ``wave`` and
        any seed values (e.g. ``sed_observed`` already populated).
    params : mapping
        Full parameter dict. Each component sees only its
        prefix-matched slice plus the bare-name allowlist.
    ssp_data : Any | None, optional
        SSP stellar population synthesis grid. When provided, is passed
        to each component's ``apply()`` method as a JIT runtime
        parameter. Components that do not need it should ignore the
        argument. Default ``None`` means components rely on their
        internal ``self.ssp_data``.
    template_data : Any | None, optional
        Nebular backend grids and weights. When provided, is passed
        to each component's ``apply()`` method as a JIT runtime
        parameter. Components that do not need it should ignore the
        argument. Default ``None`` means components rely on their
        internal template data.

    Returns
    -------
    ForwardState
        Final state after all components have applied.

    Raises
    ------
    ComponentIOError
        If, after every component has applied, the final
        ``state.derived._extras`` is non-empty — i.e. some component's
        ``apply()`` slipped data through the ``DerivedState``'s opt-in
        spillover instead of using ``state.derived.with_(X=value)``
        (ADR-0007 Phase 3). The snapshot test
        (:mod:`tests.integration.test_derived_contract_snapshots`)
        catches this for the 3 snapshotted recipes; the runtime check
        here broadens the guard to every hand-rolled component list.

        Bypass with the env var ``TENGRI_ALLOW_DERIVED_EXTRAS=1`` —
        useful during in-flight migrations or when external user code
        explicitly attaches non-canonical keys via
        ``DerivedState.from_dict(..., allow_extras=True)``. Not for
        production code.
    """
    import os as _os

    for component in components:
        sliced = slice_params_for_component(component, params)
        state = component.apply(state, sliced, ssp_data=ssp_data, template_data=template_data)

    # ADR-0007 Phase 4 invariant — strict typed-only writes (#64
    # added the same check at the snapshot-test boundary; this one
    # makes the guard apply to *every* run_components call regardless
    # of whether the pipeline is snapshotted).
    if _os.environ.get("TENGRI_ALLOW_DERIVED_EXTRAS") != "1":
        extras = getattr(state.derived, "_extras", None)
        if extras:
            raise ComponentIOError(
                f"run_components: state.derived._extras is non-empty after the "
                f"forward pass: {list(extras.keys())!r}. Some component's "
                f"apply() is writing through DerivedState's opt-in spillover "
                f"instead of the typed ``state.derived.with_(X=value)`` API "
                f"(ADR-0007 Phase 3). Migrate the offending write site to use "
                f"``with_(...)``, add a matching field to DerivedState, or "
                f"set TENGRI_ALLOW_DERIVED_EXTRAS=1 for debugging only."
            )

    return state
