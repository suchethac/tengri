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
        ``{"sf": [...], "agn": [...]}``: the additive SF and AGN radio axes.
        Use ``builders.radio.sf.<variant>()`` / ``builders.radio.agn.<variant>()``
        or hand-write ``radio={'sf': {'type': ...}, 'agn': {'type': ...}}``.
    """
    return {"sf": sf.available(), "agn": agn.available()}


# ── Legacy flat factories (back-compat) ────────────────────────────────────
# The pre-composable surface exposed ``builders.radio.<type>`` (e.g.
# ``radio.condon92``) returning a flat ``{'type': X, ...}`` dict, still used by
# the CIGALE reproduction notebook and existing tests. #1980 retired that
# spelling from ``parse_groups``, so the factories now EMIT the composable
# form: the legacy name resolves through ``_legacy_radio_type_to_blocks`` onto
# the ``sf``/``agn`` axes, and per-param overrides (``q_ir``, ``alpha_sf``, …)
# stay at the radio top level, where the composable grammar accepts them.
def _discover_legacy_params(variant: str) -> list[str]:
    if variant == "none":
        return []
    # Use the composable form to discover parameters for legacy variant
    # (legacy flat form is retired from the public API)
    from tengri.parameters.groups import _legacy_radio_type_to_blocks

    sf_variant, agn_variant = _legacy_radio_type_to_blocks(variant)
    recipe = {
        "sfh": {"type": "dpl"},
        "radio": {
            "sf": {"type": sf_variant, WILDCARD_ALIAS: FREE},
            "agn": {"type": agn_variant, WILDCARD_ALIAS: FREE},
        },
    }
    records = recipe_parameters(recipe, free_only=False)
    return [
        short_form(rec.name, prefixes=("radio_",))
        for rec in records
        if rec.name.startswith("radio_")
    ]


def _composable_output(variant: str, flat_factory: Callable[..., dict]) -> Callable[..., dict]:
    """Rewrite a flat factory's ``{'type': variant, ...}`` into the composable form.

    ``make_factory`` is shared with igm/xray and emits the ``type`` key; for
    radio that spelling is retired (#1980), so the legacy NAME survives as a
    preset that expands to its documented sf/agn resolution.
    """
    from functools import wraps

    from tengri.parameters.groups import _legacy_radio_type_to_blocks

    if variant == "none":
        sf_variant, agn_variant = "none", "none"
    else:
        sf_variant, agn_variant = _legacy_radio_type_to_blocks(variant)

    @wraps(flat_factory)
    def factory(**kwargs) -> dict:
        out = flat_factory(**kwargs)
        out.pop("type", None)
        return {"sf": {"type": sf_variant}, "agn": {"type": agn_variant}, **out}

    return factory


def _populate_legacy_factories() -> dict[str, Callable[..., dict]]:
    from tengri.parameters.groups import _valid_radio_types

    factories: dict[str, Callable[..., dict]] = {}
    for variant in sorted(_valid_radio_types()):
        factories[variant] = _composable_output(
            variant,
            make_factory(
                variant=variant,
                short_params=_discover_legacy_params(variant),
                qualname_prefix="tengri.builders.radio",
                module_name="tengri.builders.radio",
                short_doc=(f"Radio preset {variant!r} (legacy name, composable output)."),
            ),
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
