# SPDX-License-Identifier: BSD-3-Clause
"""Drift guards for PyTree-registered frozen dataclasses.

`tengri` uses :func:`jax.tree_util.register_dataclass` (JAX's native helper,
the modern replacement for the older :func:`register_pytree_node` pattern)
to make frozen dataclasses like :class:`ForwardState`, :class:`DerivedState`,
and :class:`StellarSEDComponent` traversable as JAX PyTrees.

Two failure modes this file guards against:

1. **Registration drift on a registered class.** Someone adds a new field to
   a registered dataclass but forgets to update the ``data_fields`` /
   ``meta_fields`` tuple in the matching ``register_dataclass`` call. JAX
   actually catches this at import time (raises ``ValueError``), but a
   targeted round-trip test makes the failure mode loud and locates the
   regression precisely.

2. **Hidden array field in an unregistered subclass.** ``SEDComponentState``,
   ``SEDComponentConfig``, and ``DerivedState`` have ~29 subclasses
   (per-component config / state types). These subclasses are intentionally
   *not* registered as PyTrees — they're held in the parent's ``meta_fields``
   and used as static metadata. But if someone adds a traced array field to
   such a subclass thinking JAX will see it, the field silently disappears
   from the trace (whole instance becomes one opaque leaf). The guard below
   flags any non-static-looking field on those subclasses so the author
   knows to register the subclass explicitly.

These tests sit in ``tests/contract/`` because they encode a contract about
the JAX-pytree representation of public types — not a physics behaviour.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pkgutil
from collections.abc import Callable

import jax.numpy as jnp
import pytest
from jax import tree_util as tu

import tengri
from tengri.protocols.component import (
    ForwardState,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.protocols.derived_state import DerivedState

pytestmark = pytest.mark.contract

# ── (1) Registered classes round-trip cleanly ─────────────────────────────

# (cls, factory that builds a representative instance)
_REGISTERED_CASES: list[tuple[type, Callable[[], object]]] = [
    (
        ForwardState,
        lambda: ForwardState(
            wave=jnp.array([1.0, 2.0]),
            sed_intrinsic=jnp.array([3.0, 4.0]),
            sed_attenuated=jnp.array([5.0, 6.0]),
            sed_observed=jnp.array([7.0, 8.0]),
            lines={"halpha": jnp.array([9.0, 10.0])},
            derived=DerivedState(log_mstar=jnp.asarray(10.5)),
        ),
    ),
    (SEDComponentState, lambda: SEDComponentState(name="probe")),
    (SEDComponentConfig, lambda: SEDComponentConfig(name="probe")),
    (DerivedState, lambda: DerivedState(log_mstar=jnp.asarray(10.5))),
]


@pytest.mark.parametrize(
    "cls,factory",
    _REGISTERED_CASES,
    ids=[c.__name__ for c, _ in _REGISTERED_CASES],
)
def test_registered_dataclass_roundtrip(cls, factory):
    """Every field of a registered dataclass survives PyTree flatten + unflatten."""
    instance = factory()
    leaves, treedef = tu.tree_flatten(instance)
    rebuilt = tu.tree_unflatten(treedef, leaves)
    assert type(rebuilt) is type(instance)
    for field in dataclasses.fields(instance):
        orig = getattr(instance, field.name)
        new = getattr(rebuilt, field.name)
        if isinstance(orig, jnp.ndarray):
            assert jnp.array_equal(orig, new), (
                f"{cls.__name__}.{field.name} not preserved by PyTree round-trip — "
                f"check the matching jax.tree_util.register_dataclass call covers it"
            )
        elif isinstance(orig, dict):
            # Dict fields (e.g. ForwardState.lines): JAX's dict-pytree handler
            # recurses; just confirm keys round-trip
            assert set(new.keys()) == set(orig.keys())
        else:
            assert new == orig, (
                f"{cls.__name__}.{field.name} not preserved by PyTree round-trip "
                f"(got {new!r}, expected {orig!r})"
            )


# ── (2) Unregistered subclasses must not carry traced array fields ────────


def _walk_tengri_classes() -> list[type]:
    """Walk every importable module under ``tengri`` and collect dataclass classes."""
    seen: set[type] = set()
    out: list[type] = []
    for _, modname, _ in pkgutil.walk_packages(tengri.__path__, prefix="tengri."):
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if obj in seen or not dataclasses.is_dataclass(obj):
                continue
            seen.add(obj)
            out.append(obj)
    return out


# These field names are known-OK on unregistered subclasses because they are
# either Python scalars / strings (config) or non-traced callables / paths
# (state). Anything else looks suspicious.
_STATIC_FIELD_TYPES: tuple[type, ...] = (str, int, float, bool, type(None), Callable, type)


def _is_clearly_static(annotation_str: str) -> bool:
    """Heuristic: a field is 'clearly static' if its annotation suggests a
    Python scalar, string, callable, path, or another *Config / *State /
    Protocol type (all of which are themselves static-meta in our convention).
    Errs on the side of letting fields through — the test only flags
    obvious tensor-looking fields on unregistered subclasses."""
    s = annotation_str.lower()
    static_markers = (
        "str",
        "int",
        "float",
        "bool",
        "callable",
        "path",
        "config",
        "state",
        "protocol",
        "literal",
        "tuple",
        "mapping[str",
        "dict[str",
        "frozenset",
        "set[",
    )
    return any(m in s for m in static_markers)


# Allowlist of currently-known ``ClassName.field_name`` pairs where a
# subclass of a registered dataclass carries an array-typed annotation but
# is *intentionally* held as static metadata in its parent's registration
# (so JAX never traces these fields). Verified at the time of writing;
# new additions to this allowlist need a reviewer to confirm that the
# field is never passed into a traced JAX context.
_KNOWN_STATIC_ARRAY_FIELDS: frozenset[str] = frozenset(
    {
        "GRAHSPSEDComponentState.feii_wave_nm",
        "GRAHSPSEDComponentState.feii_lumin",
        "GRAHSPSEDComponentState.line_wave_nm",
        "GRAHSPSEDComponentState.line_broad",
        "GRAHSPSEDComponentState.line_narrow_sy2",
        "GRAHSPSEDComponentState.line_narrow_liner",
        # GRAHSP parity port: Veron-Cetty FeII, MN12 template torus, Netzer disc
        # template grids — static precompute-time tensors (same status as above).
        "GRAHSPSEDComponentState.feii_vc04_wave_nm",
        "GRAHSPSEDComponentState.feii_vc04_lumin",
        "GRAHSPSEDComponentState.torus_mn12_wave_nm",
        "GRAHSPSEDComponentState.torus_mn12_avg",
        "GRAHSPSEDComponentState.torus_mn12_lo",
        "GRAHSPSEDComponentState.torus_mn12_hi",
        "GRAHSPSEDComponentState.torus_mn12_si_wave_nm",
        "GRAHSPSEDComponentState.torus_mn12_si_lumin",
        "GRAHSPSEDComponentState.disc_wave_nm",
        "GRAHSPSEDComponentState.disc_lumin",
        "DustEmissionSEDComponentState.pahspec_lgU_grid",
        "DustEmissionSEDComponentState.pahspec_lnu_template",
        "DustEmissionSEDComponentState.pahspec_norm_per_lgU",
        "DustEmissionSEDComponentState.astrodust_lgU_grid",
        "DustEmissionSEDComponentState.astrodust_lnu_template",
        "DustEmissionSEDComponentState.astrodust_norm_per_lgU",
        "DustEmissionSEDComponentState.astrodust_lnu_spinning",
        "DustAttenuationSEDComponentState.k_lambda",
    }
)


def test_unregistered_subclasses_carry_only_static_fields():
    """Subclasses of registered dataclasses that themselves go unregistered
    must not declare *new* traced array fields without registration.

    A subclass that needs a traced field has to call
    :func:`jax.tree_util.register_dataclass` for itself — otherwise JAX
    flattens the parent type and silently drops the new field from the
    trace, which is the worst kind of regression (no error, just wrong
    behaviour under JIT / grad / vmap).

    The classes / fields listed in ``_KNOWN_STATIC_ARRAY_FIELDS`` are the
    snapshot of cases at the time of writing — all verified to be held as
    static meta in their parent component's registration. The test only
    fails on *new* additions. If you add a new state subclass with array
    fields, you must either:

    1. Register the subclass with ``register_dataclass`` (preferred when
       the field is traced — e.g. depends on parameters), OR
    2. Add the new ``ClassName.field`` to the allowlist (when the field
       is genuinely static — e.g. a fixed template grid set once at
       precompute time and never re-derived).

    Reviewers: option 2 needs explicit justification in the PR description.
    """
    import jax._src.tree_util as ptu

    registered = ptu._registry
    parents = (SEDComponentState, SEDComponentConfig, ForwardState, DerivedState)
    parent_field_names = {P: {f.name for f in dataclasses.fields(P)} for P in parents}

    new_suspicious: list[str] = []
    for cls in _walk_tengri_classes():
        if cls in parents or cls in registered:
            continue
        parent_hits = [P for P in parents if issubclass(cls, P)]
        if not parent_hits:
            continue
        own_fields = [
            f for f in dataclasses.fields(cls) if f.name not in parent_field_names[parent_hits[0]]
        ]
        for f in own_fields:
            ann_str = str(f.type) if not isinstance(f.type, str) else f.type
            if "ndarray" in ann_str.lower() or "array" in ann_str.lower():
                if _is_clearly_static(ann_str):
                    continue
                key = f"{cls.__qualname__}.{f.name}"
                if key not in _KNOWN_STATIC_ARRAY_FIELDS:
                    new_suspicious.append(
                        f"{cls.__module__}.{cls.__qualname__}.{f.name}: {ann_str}"
                    )

    if new_suspicious:
        msg = (
            "New unregistered dataclass subclass(es) carry array-typed fields. "
            "Either:\n"
            "  1. Register the subclass via "
            "``jax.tree_util.register_dataclass`` (preferred when the field "
            "is traced), OR\n"
            "  2. Add ``ClassName.field`` to ``_KNOWN_STATIC_ARRAY_FIELDS`` "
            "in this file (only when the field is genuinely static "
            "metadata — e.g. a precompute-time template grid).\n\n"
            "New additions detected:\n" + "\n".join(f"  - {s}" for s in new_suspicious)
        )
        pytest.fail(msg)


def test_allowlist_only_contains_real_fields():
    """If a class or field listed in the allowlist gets renamed or removed,
    we want the allowlist to fail-loud rather than silently mask future drift."""
    all_subclass_fields: set[str] = set()
    for cls in _walk_tengri_classes():
        if not dataclasses.is_dataclass(cls):
            continue
        for f in dataclasses.fields(cls):
            all_subclass_fields.add(f"{cls.__qualname__}.{f.name}")

    stale = _KNOWN_STATIC_ARRAY_FIELDS - all_subclass_fields
    if stale:
        pytest.fail(
            "These entries in _KNOWN_STATIC_ARRAY_FIELDS no longer exist in the "
            "codebase (class or field was renamed / removed). Update the allowlist:\n"
            + "\n".join(f"  - {s}" for s in sorted(stale))
        )
