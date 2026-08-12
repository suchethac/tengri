# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories for nebular emission config dicts.

Variants come from :data:`tengri.components.nebular._models.NEBULAR_MODELS`:

- ``none`` — no nebular contribution.
- ``ssp`` — embedded in the SSP grid (no free params; selects the SSP
  convention).
- ``cue`` — neural emulator (Li+ 2024).
- ``cloudy`` — direct CLOUDY interface (requires a Cloudy grid path).
- ``cb19`` — Charlot & Bruzual 2019 templates.

The free-parameter set is **shared** between ``cloudy`` / ``cue`` /
``cb19`` (per :mod:`tengri.parameters._builders`), with ``cb19``
adding a handful of extras. ``none`` and ``ssp`` carry no free
parameters.

Auto-discovery is via :func:`recipe_parameters`. The ``cloudy``
variant cannot be introspected without a Cloudy grid file on disk, so
its short-param list is taken from ``cue`` (the parser activates an
identical set for both backends).

Examples
--------
>>> from tengri import builders, FREE, Uniform
>>> builders.neb.cue(defaults=FREE, fesc=Uniform(0.0, 0.5))  # doctest: +SKIP
{'type': 'cue', 'all_params': FREE, 'fesc': Uniform(...)}
"""

from __future__ import annotations

from collections.abc import Callable

from tengri._completion import curated_dir
from tengri.builders._factory import make_factory, short_form
from tengri.parameters.groups import _valid_nebular_types
from tengri.parameters.registry import recipe_parameters
from tengri.parameters.sentinels import FREE, WILDCARD_ALIAS

_PREFIXES = ("neb_", "ionspec_", "gas_log")


def _discover_params(variant: str) -> list[str]:
    """Return the short-form free-param names this variant activates.

    Returns an empty list for ``none`` / ``ssp`` (zero free params).
    For ``cloudy`` (which needs a Cloudy grid path the parser can't
    fabricate during introspection), falls back to the ``cue`` set —
    they share ``_NEBULAR_PARAMS`` in the param registry.
    """
    if variant in {"none", "ssp"}:
        return []
    introspect_variant = "cue" if variant == "cloudy" else variant
    recipe = {
        "sfh": {"type": "dpl"},
        "neb": {"type": introspect_variant, WILDCARD_ALIAS: FREE},
    }
    records = recipe_parameters(recipe, free_only=False)
    return [
        short_form(rec.name, prefixes=_PREFIXES)
        for rec in records
        if any(rec.name.startswith(p) for p in _PREFIXES)
    ]


def _populate_factories() -> dict[str, Callable[..., dict]]:
    factories: dict[str, Callable[..., dict]] = {}
    for variant in sorted(_valid_nebular_types()):
        factories[variant] = make_factory(
            variant=variant,
            short_params=_discover_params(variant),
            qualname_prefix="tengri.builders.neb",
            module_name="tengri.builders.neb",
            short_doc=f"Nebular emission backend: {variant!r}.",
        )
    return factories


_FACTORIES = _populate_factories()
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of nebular variant names exposed by this module."""
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]


__dir__ = curated_dir(__all__)
