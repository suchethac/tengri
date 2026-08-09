# SPDX-License-Identifier: BSD-3-Clause
"""Every per-galaxy channel is declared once and checked against every engine.

``Catalog`` writes per-galaxy data onto a dict; three engines read it -- the
sequential per-galaxy ``Fitter`` loop, the vmapped MCMC engine, and the batched
native-VI engine. The knowledge of *which* channels exist used to live on both
sides at once: ``catalog.py`` wrote keys, and each engine read a hand-picked
subset. A channel added to one side and not the other was not refused, it was
dropped in silence, because an engine that never reads a key raises nothing.

That single shape produced #1460 (limits lost at the ``Data`` seam), #1480
(per-galaxy lines needed at all) and #1599 (per-galaxy lines honored by
``mcmc_nuts`` and ignored by ``map``, the default).

:data:`GALAXY_CHANNELS` is now the one declaration, and
``_refuse_unsupported_channels`` the one rule. These tests pin the two
directions that let the halves drift apart.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from tengri.inference import catalog as catalog_mod
from tengri.inference.catalog_fitter import (
    _ALL_ENGINES,
    GALAXY_CHANNELS,
    CatalogFitter,
)

pytestmark = pytest.mark.contract


def _keys_written_by_catalog() -> set[str]:
    """Galaxy-dict keys ``catalog.py`` assigns, read off the AST.

    Matches ``galaxy_dict["<literal>"] = ...`` and the dict literal the
    variable is initialized from. AST rather than a regex so a commented-out
    or string-embedded assignment cannot register as a real one.
    """
    tree = ast.parse(Path(inspect.getfile(catalog_mod)).read_text())
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            # galaxy_dict["key"] = ...
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "galaxy_dict"
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                keys.add(target.slice.value)
            # galaxy_dict = {"key": ..., ...}
            if (
                isinstance(target, ast.Name)
                and target.id == "galaxy_dict"
                and isinstance(node.value, ast.Dict)
            ):
                for k in node.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        keys.add(k.value)
    return keys


def test_every_key_catalog_writes_is_a_declared_channel():
    """The producing side may not invent a channel the rule does not know.

    This is the direction that actually bit: ``catalog.py`` started writing
    ``line_flux_obs``, the sequential engine never read it, and nothing said
    so for as long as the default method was ``map``.
    """
    written = _keys_written_by_catalog()
    declared = {key for ch in GALAXY_CHANNELS for key in ch.keys}

    undeclared = written - declared
    assert not undeclared, (
        f"catalog.py writes per-galaxy key(s) {sorted(undeclared)} that "
        "GALAXY_CHANNELS does not declare. An engine that does not read them "
        "drops them in silence — declare each one with the engines that "
        "actually thread it."
    )


def test_every_declared_channel_is_actually_produced():
    """And the rule may not describe a channel nothing produces.

    A stale row would refuse a method for a channel that can never appear,
    which is a different way for the table to stop describing reality.
    """
    written = _keys_written_by_catalog()
    declared = {key for ch in GALAXY_CHANNELS for key in ch.keys}

    orphaned = declared - written
    assert not orphaned, (
        f"GALAXY_CHANNELS declares key(s) {sorted(orphaned)} that catalog.py "
        "never writes — the table has drifted from the producer."
    )


def test_channel_engines_are_known_engine_kinds():
    """A typo'd engine name would silently make a channel unsupported."""
    for channel in GALAXY_CHANNELS:
        unknown = channel.engines - _ALL_ENGINES
        assert not unknown, (
            f"channel {channel.name!r} names unknown engine(s) "
            f"{sorted(unknown)}; valid: {sorted(_ALL_ENGINES)}"
        )
        assert channel.engines, f"channel {channel.name!r} names no engine at all"


def test_photometry_is_carried_by_every_engine():
    """The one channel with no valid refusal.

    If flux/noise were ever marked unsupported the rule would refuse every
    catalog fit, so this pins the row that must stay universal.
    """
    photometry = next(ch for ch in GALAXY_CHANNELS if ch.name == "photometry")
    assert photometry.engines == _ALL_ENGINES


@pytest.mark.parametrize(
    "method,expected",
    [
        ("map", "sequential"),
        ("mcmc_nuts", "mcmc_vmapped"),
        ("mcmc_hmc", "mcmc_vmapped"),
        ("native_vi_linear", "native_vi"),
        ("native_vi_nonlinear", "native_vi"),
    ],
)
def test_engine_kind_classifies_each_method(method, expected):
    """The rule is only correct if it maps methods to the right engine."""
    fitter = CatalogFitter.__new__(CatalogFitter)
    assert fitter._engine_kind(method) == expected
