# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories for SFH config dicts.

Each function in this module corresponds to one SFH variant registered in
:data:`tengri.components.stellar.sfh.registry.SFH_REGISTRY` (the
single source of truth for SFH model variants and their fittable
parameters). Calling a factory returns a plain :class:`dict` matching the
nested-dict grammar consumed by :meth:`tengri.SEDModel.build` and
:func:`tengri.parse_groups`, so factories and the dict path are
freely interchangeable.

Why factories rather than dicts
-------------------------------
The dict form has no IDE autocomplete on inner parameter names: typos
like ``{'beta_': Uniform(1, 3)}`` surface only at construction time. The
factories give per-variant signatures generated from the registry, so:

- Hovering over ``builders.sfh.dpl`` in an IDE shows the real parameter
  list (``alpha``, ``beta``, ``tau_gyr``, ``log_total_mass``).
- A typo (``beat=Uniform(1,3)``) is rejected immediately with a
  :class:`TypeError` listing valid parameter names.
- The registry remains the canonical source; factories regenerate
  automatically when variants are added or their parameters change.

Examples
--------
>>> from tengri import SEDModel, builders, FREE, DEFAULT, Uniform, Fixed
>>> # Equivalent to {'type': 'dpl', 'beta': Uniform(1, 3), 'other_params': Fixed(DEFAULT)}
>>> sfh_config = builders.sfh.dpl(beta=Uniform(1, 3))
>>> # All params free unless overridden:
>>> sfh_config = builders.sfh.dpl(all_params=FREE)
>>> # Mix wildcard policy with explicit overrides:
>>> # (emits {'type': 'dpl', 'log_total_mass': Fixed(10.0), 'other_params': FREE})
>>> sfh_config = builders.sfh.dpl(all_params=FREE, log_total_mass=Fixed(10.0))

The output is interchangeable with the dict form:

>>> model = SEDModel.build(ssp_data=ssp, sfh=builders.sfh.dpl(all_params=FREE))
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from tengri._completion import curated_dir
from tengri.builders._factory import _DEFAULT_WILDCARD, _pop_wildcard, _validate_wildcard
from tengri.components.stellar.sfh.registry import SFH_REGISTRY
from tengri.parameters.sentinels import WILDCARD_ALIAS, WILDCARD_ALIAS_OTHER

# Sentinel marking a parameter that was not specified at call time. We
# can't use ``None`` because ``None`` is a legitimate dict value users
# might want to pass through; can't use ``FREE`` / ``Fixed(DEFAULT)`` because
# those carry semantic meaning. A bare ``object()`` does the job.
_UNSET = object()


def _short_form(full_param_name: str) -> str:
    """Mirror ``tengri.parameters.groups._extract_short_name`` for SFH.

    The grammar's parser strips ``sfh_`` then splits on the first
    underscore, so ``sfh_dpl_alpha`` becomes ``alpha`` and
    ``sfh_cexp_log_total_mass`` becomes ``log_sfr``. Factories must mirror this
    rule so their kwargs match what the parser will look up.
    """
    if not full_param_name.startswith("sfh_"):
        return full_param_name
    rest = full_param_name[4:]
    parts = rest.split("_", 1)
    return parts[1] if len(parts) == 2 else rest


def _build_docstring(variant: str, spec, param_records: list[tuple[str, Any]]) -> str:
    """Compose a numpydoc-style docstring from the registry entry."""
    entry = SFH_REGISTRY[variant]
    short_doc = getattr(entry, "short_doc", "") or getattr(spec, "short_doc", "")
    citation = getattr(entry, "citation", "")
    composition = getattr(spec, "composition_type", "")

    lines: list[str] = []
    lines.append(f"Build a config dict for the {variant!r} SFH variant.")
    lines.append("")
    if short_doc:
        lines.append(short_doc)
        lines.append("")
    lines.append("Parameters")
    lines.append("----------")
    lines.append("all_params : sentinel or Fixed, optional")
    lines.append(
        "    Wildcard policy for parameters not explicitly named in this call. "
        "``FREE`` makes them fit; ``Fixed(DEFAULT)`` (default) pins them to their "
        "registry-default center. Matches the ``'all_params'`` key in the dict grammar."
    )
    lines.append("other_params : sentinel, optional")
    lines.append(
        "    Exact synonym of ``all_params``; give only one. Reads best written "
        'last, after explicit per-parameter overrides, as "the others". Matches '
        "the ``'other_params'`` key in the dict grammar."
    )
    for short, pdef in param_records:
        default_repr = repr(pdef.default) if pdef.default is not None else "registry default"
        lines.append(f"{short} : Distribution, sentinel, or scalar, optional")
        lines.append(f"    {pdef.description}. Default prior: {default_repr}.")
    lines.append("")
    lines.append("Returns")
    lines.append("-------")
    lines.append("dict")
    lines.append(
        "    Config dict matching the :meth:`SEDModel.build` grammar; pass as "
        f"``sfh={variant}(...)`` or store and reuse."
    )
    if citation or composition:
        lines.append("")
        lines.append("Notes")
        lines.append("-----")
        if composition:
            lines.append(f"Composition type: {composition}.")
        if citation:
            lines.append(f"Citation: {citation}.")
    return "\n".join(lines)


def _make_factory(variant: str, spec) -> Callable[..., dict]:
    """Build one factory callable for a single SFH variant.

    The returned function carries a synthetic :class:`inspect.Signature`
    listing the wildcard kwarg ``_`` followed by one keyword-only entry
    per short-form parameter name. Modern IDEs (PyCharm, Pylance) honor
    ``__signature__`` for autocomplete on call sites.
    """
    param_records: list[tuple[str, Any]] = []
    for full_name, pdef in spec.params.items():
        short = _short_form(full_name)
        param_records.append((short, pdef))
    short_names = [s for s, _ in param_records]

    def factory(**kwargs: Any) -> dict:
        wildcard = _pop_wildcard(variant, kwargs)
        _validate_wildcard(variant, wildcard)
        unknown = [k for k in kwargs if k not in short_names]
        if unknown:
            raise TypeError(
                f"{variant}() got unexpected keyword arguments: {unknown}. "
                f"Valid parameter names for {variant!r}: {short_names}. "
                f"(Pass ``all_params=FREE`` or ``all_params=Fixed(DEFAULT)`` -- or the "
                f"synonym ``other_params=`` -- to set the policy.)"
            )
        out: dict[str, Any] = {"type": variant}
        # Per-parameter entries before the wildcard.
        param_entries: dict[str, Any] = {
            short: kwargs[short]
            for short in short_names
            if short in kwargs and kwargs[short] is not _UNSET
        }
        out.update(param_entries)
        # Wildcard LAST: 'all_params' when it is the only parameter
        # directive, 'other_params' when explicit per-param entries precede it.
        wildcard_key = WILDCARD_ALIAS if not param_entries else WILDCARD_ALIAS_OTHER
        out[wildcard_key] = wildcard
        return out

    # Real signature so IDEs see per-parameter kwargs.
    sig_params = [
        inspect.Parameter(
            "all_params",
            inspect.Parameter.KEYWORD_ONLY,
            default=_DEFAULT_WILDCARD,
            annotation=Any,
        ),
        inspect.Parameter(
            "other_params",
            inspect.Parameter.KEYWORD_ONLY,
            default=_UNSET,
            annotation=Any,
        ),
    ]
    for short, _pdef in param_records:
        sig_params.append(
            inspect.Parameter(
                short,
                inspect.Parameter.KEYWORD_ONLY,
                default=_UNSET,
                annotation=Any,
            )
        )
    factory.__signature__ = inspect.Signature(sig_params, return_annotation=dict)
    factory.__name__ = variant
    factory.__qualname__ = f"tengri.builders.sfh.{variant}"
    factory.__module__ = "tengri.builders.sfh"
    factory.__doc__ = _build_docstring(variant, spec, param_records)
    return factory


def _populate_factories() -> dict[str, Callable[..., dict]]:
    """Walk SFH_REGISTRY and emit one factory per canonical variant name.

    Short-name aliases (e.g. ``tsnorm`` for ``truncated_skewnormal``) are
    skipped: we emit one factory per spec, keyed by the spec's canonical name.
    Same convention as the dict grammar's primary keys.
    """
    from tengri.components.stellar.sfh.registry import UNVALIDATED_SFH_TYPES

    factories: dict[str, Callable[..., dict]] = {}
    seen: set[int] = set()
    for key, entry in SFH_REGISTRY.items():
        # Skip SFHs not yet validated against the DSPS forward path: the
        # grammar rejects them, so emitting a factory would produce a callable
        # that errors at build (advertised-but-unusable). See UNVALIDATED_SFH_TYPES.
        if key in UNVALIDATED_SFH_TYPES:
            continue
        spec = entry.callable if hasattr(entry, "callable") else entry
        if id(spec) in seen:
            continue
        # Skip alias keys: only emit under the canonical name.
        if getattr(spec, "name", None) != key:
            continue
        seen.add(id(spec))
        factories[key] = _make_factory(key, spec)
    return factories


_FACTORIES = _populate_factories()

# Promote each factory to a module-level attribute so users can call
# ``builders.sfh.dpl(...)``. Doing this via ``globals().update`` (rather
# than emitting one ``def`` per variant) keeps the registry as the single
# source of truth: adding or removing a variant in ``SFH_REGISTRY``
# automatically reflects here.
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the SFH variant names that can actually be built.

    The canonical keys of :data:`SFH_REGISTRY` **minus**
    :data:`~tengri.components.stellar.sfh.registry.UNVALIDATED_SFH_TYPES`:
    types that are registered but not yet wired into the DSPS forward path, so
    ``SEDModel.build`` raises on them. Aliases are not surfaced; call them via
    the canonical name.

    Returns
    -------
    list of str
        Buildable variant names, sorted.

    Notes
    -----
    This is deliberately shorter than :func:`tengri.list_sfh_models`, which
    reports every *registered* type; including the unvalidated ones, marked
    ``status='unvalidated'``: so the two answer different questions: "what can
    I build?" versus "what exists?". The counts differ by exactly the
    unvalidated set.

    Examples
    --------
    >>> from tengri import builders
    >>> "dpl" in builders.sfh.available()
    True
    """
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]


__dir__ = curated_dir(__all__)
