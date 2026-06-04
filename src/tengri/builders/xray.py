# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories for X-ray config dicts.

Variants come from :data:`tengri.parameters.groups._VALID_XRAY_TYPES`
(currently ``none`` and ``simple``). Like :mod:`tengri.builders.radio`,
the free-parameter set is shared across variants — the variant string
selects the physics model.

.. note::
   See :mod:`tengri.builders.radio` for the same wildcard caveat: the
   parser does not currently flip individual X-ray params to FREE via
   the ``*`` wildcard. Use per-param :class:`Distribution` overrides
   to make X-ray params free (e.g.
   ``builders.xray.simple(alpha_ox=Uniform(-2.0, -1.0))``).

Examples
--------
>>> from tengri import builders, Uniform
>>> builders.xray.simple(alpha_ox=Uniform(-2.0, -1.0))  # doctest: +SKIP
{'type': 'simple', '*': FIXED, 'alpha_ox': Uniform(...)}
"""

from __future__ import annotations

from collections.abc import Callable

from tengri.builders._factory import make_factory, short_form
from tengri.parameters.groups import _valid_xray_types
from tengri.parameters.registry import recipe_parameters
from tengri.parameters.sentinels import FREE

_PREFIXES = ("xray_",)


def _discover_params(variant: str) -> list[str]:
    if variant == "none":
        return []
    recipe = {
        "sfh": {"type": "dpl"},
        "xray": {"type": variant, "*": FREE},
    }
    records = recipe_parameters(recipe, free_only=False)
    return [
        short_form(rec.name, prefixes=_PREFIXES)
        for rec in records
        if rec.name.startswith(_PREFIXES)
    ]


def _populate_factories() -> dict[str, Callable[..., dict]]:
    factories: dict[str, Callable[..., dict]] = {}
    for variant in sorted(_valid_xray_types()):
        factories[variant] = make_factory(
            variant=variant,
            short_params=_discover_params(variant),
            qualname_prefix="tengri.builders.xray",
            module_name="tengri.builders.xray",
            short_doc=f"X-ray model: {variant!r}.",
        )
    return factories


_FACTORIES = _populate_factories()
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of X-ray variant names exposed by this module."""
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]
