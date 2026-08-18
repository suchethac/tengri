# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories for radio config dicts with composable SF + AGN sub-blocks.

Radio physics is decomposed into two independent axes:

**SF synchrotron** (FIR–radio correlation):

- ``none``: SF radio turned off (AGN only)
- ``bell2003``: fixed q_IR (default)
- ``delvecchio2021``: mass + redshift dependent at 1.4 GHz
- ``mccheyne2022``: mass + redshift dependent at 150 MHz

**AGN radio** (jets + lobes):

- ``none``: AGN radio turned off (SF only)
- ``powerlaw``: single power-law (default)
- ``dpl``: double power-law with aging cutoff (AGNfitter-rx)

The nested-dict grammar supports independent selection:

.. code-block:: python

    radio = {
        "sf": {"type": "delvecchio2021"},  # SF variant
        "agn": {"type": "dpl"},  # AGN variant
    }

Each sub-block can carry variant-specific or shared radio parameters
(e.g. ``q_ir``, ``radio_loudness``).

Examples
--------
>>> from tengri import builders, Uniform
>>> radio = {
...     "sf": {"type": "delvecchio2021"},
...     "agn": {"type": "dpl"},
... }
>>> radio = builders.radio.sf.delvecchio2021(delv_q0=Uniform(2.4, 3.1))  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Callable

from tengri._completion import curated_dir
from tengri.builders._factory import make_factory, short_form
from tengri.components.radio.component import AGN_RADIO_MODELS
from tengri.parameters.registry import recipe_parameters
from tengri.parameters.sentinels import FREE, WILDCARD_ALIAS

# ── SF synchrotron axis ────────────────────────────────────────────────────

_SF_VARIANTS = frozenset({"none", "bell2003", "delvecchio2021", "mccheyne2022"})


def _discover_sf_params(variant: str) -> list[str]:
    """Discover short-form param names for a given SF variant."""
    if variant == "none":
        return []
    recipe = {
        "sfh": {"type": "dpl"},
        "radio": {"sf": {"type": variant, WILDCARD_ALIAS: "FREE_PLACEHOLDER"}},
    }

    recipe["radio"]["sf"][WILDCARD_ALIAS] = FREE
    records = recipe_parameters(recipe, free_only=False)
    return [
        short_form(rec.name, prefixes=("radio_",))
        for rec in records
        if rec.name.startswith("radio_")
    ]


def _populate_sf_factories() -> dict[str, Callable[..., dict]]:
    """Build one factory per SF variant."""
    factories: dict[str, Callable[..., dict]] = {}
    for variant in sorted(_SF_VARIANTS):
        factories[variant] = make_factory(
            variant=variant,
            short_params=_discover_sf_params(variant),
            qualname_prefix="tengri.builders.radio.sf",
            module_name="tengri.builders.radio.sf",
            short_doc=f"Radio SF model: {variant!r}.",
        )
    return factories


class SFNamespace:
    """Namespace for SF synchrotron factories."""

    def __init__(self, factories: dict[str, Callable[..., dict]]):
        self._factories = factories
        for name, factory in factories.items():
            setattr(self, name, factory)

    def available(self) -> list[str]:
        """Return the list of SF variants."""
        return sorted(self._factories)

    def __all__(self) -> list[str]:
        return ["available", *sorted(self._factories)]


_SF_FACTORIES = _populate_sf_factories()
sf = SFNamespace(_SF_FACTORIES)

# ── AGN radio axis ────────────────────────────────────────────────────────


def _discover_agn_params(variant: str) -> list[str]:
    """Discover short-form param names for a given AGN variant."""
    if variant == "none":
        return []
    recipe = {
        "sfh": {"type": "dpl"},
        "radio": {"agn": {"type": variant, WILDCARD_ALIAS: "FREE_PLACEHOLDER"}},
    }

    recipe["radio"]["agn"][WILDCARD_ALIAS] = FREE
    records = recipe_parameters(recipe, free_only=False)
    return [
        short_form(rec.name, prefixes=("radio_",))
        for rec in records
        if rec.name.startswith("radio_")
    ]


def _populate_agn_factories() -> dict[str, Callable[..., dict]]:
    """Build one factory per AGN variant."""
    factories: dict[str, Callable[..., dict]] = {}
    for variant in sorted(AGN_RADIO_MODELS):
        factories[variant] = make_factory(
            variant=variant,
            short_params=_discover_agn_params(variant),
            qualname_prefix="tengri.builders.radio.agn",
            module_name="tengri.builders.radio.agn",
            short_doc=f"Radio AGN model: {variant!r}.",
        )
    return factories


class AGNNamespace:
    """Namespace for AGN radio factories."""

    def __init__(self, factories: dict[str, Callable[..., dict]]):
        self._factories = factories
        for name, factory in factories.items():
            setattr(self, name, factory)

    def available(self) -> list[str]:
        """Return the list of AGN variants."""
        return sorted(self._factories)

    def __all__(self) -> list[str]:
        return ["available", *sorted(self._factories)]


_AGN_FACTORIES = _populate_agn_factories()
agn = AGNNamespace(_AGN_FACTORIES)

# ── Top-level interface ────────────────────────────────────────────────────


def axes() -> dict[str, list[str]]:
    """Return the composable radio axes and their variants.

    Returns
    -------
    dict
        ``{"sf": [...], "agn": [...]}`` — the additive SF and AGN radio axes.
        Use ``builders.radio.sf.<variant>()`` / ``builders.radio.agn.<variant>()``
        or hand-write ``radio={'sf': {'type': ...}, 'agn': {'type': ...}}``.
    """
    return {"sf": sf.available(), "agn": agn.available()}


# ── Legacy flat factories (back-compat) ────────────────────────────────────
# The pre-composable surface exposed ``builders.radio.<type>`` (e.g.
# ``radio.condon92``) returning a flat ``{'type': X, ...}`` dict — still used by
# the CIGALE reproduction notebook and existing tests. The legacy ``type`` form
# remains accepted by ``parse_groups`` (radio on with default sf/agn models), so
# these factories coexist with the new ``sf``/``agn`` axes.
def _discover_legacy_params(variant: str) -> list[str]:
    if variant == "none":
        return []
    recipe = {"sfh": {"type": "dpl"}, "radio": {"type": variant, WILDCARD_ALIAS: FREE}}
    records = recipe_parameters(recipe, free_only=False)
    return [
        short_form(rec.name, prefixes=("radio_",))
        for rec in records
        if rec.name.startswith("radio_")
    ]


def _populate_legacy_factories() -> dict[str, Callable[..., dict]]:
    from tengri.parameters.groups import _valid_radio_types

    factories: dict[str, Callable[..., dict]] = {}
    for variant in sorted(_valid_radio_types()):
        factories[variant] = make_factory(
            variant=variant,
            short_params=_discover_legacy_params(variant),
            qualname_prefix="tengri.builders.radio",
            module_name="tengri.builders.radio",
            short_doc=f"Radio model (legacy flat form): {variant!r}.",
        )
    return factories


_LEGACY_FACTORIES = _populate_legacy_factories()
globals().update(_LEGACY_FACTORIES)


def available() -> list[str]:
    """Return the legacy flat radio variant names (e.g. ``condon92``, ``none``).

    Parallel with ``builders.igm.available()`` / ``builders.xray.available()``.
    For the composable SF/AGN axes use :func:`axes` (or ``radio.sf.available()`` /
    ``radio.agn.available()``).
    """
    return sorted(_LEGACY_FACTORIES)


__all__ = ["agn", "available", "axes", "sf", *sorted(_LEGACY_FACTORIES)]


__dir__ = curated_dir(__all__)
