# SPDX-License-Identifier: BSD-3-Clause
"""Preset registry for self-describing model configurations.

Presets are callable factories that return (SEDModel, Parameters) tuples,
enabling users to fit with one import. This module implements the registry
pattern matching the design in src/tengri/registry.py.

Use:
    from tengri.presets import list_presets, describe_preset, synthesizer_default

    model, params = synthesizer_default()
    presets = list_presets()  # table of all available presets
    describe_preset("synthesizer_default")  # full metadata
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tengri.registry import _RegistryTable


@dataclass(frozen=True)
class PresetEntry:
    """Registry entry for a preset factory.

    Parameters
    ----------
    name : str
        Unique identifier (e.g., "synthesizer_default").
    callable : Callable
        Factory function returning (model, params) tuple.
    short_doc : str
        One-line description.
    citations : list[str]
        BibTeX keys from docs/dev/synthesizer_parity_citations.md.
    status : str
        One of "stable", "experimental", "deprecated".
    """

    name: str
    callable: Callable
    short_doc: str
    citations: list[str]
    status: str = "stable"


_PRESET_REGISTRY: dict[str, PresetEntry] = {}


def register_preset(
    name: str,
    *,
    short_doc: str,
    citations: list[str],
    status: str = "stable",
):
    """Decorator to register a preset factory.

    Parameters
    ----------
    name : str
        Unique identifier.
    short_doc : str
        One-line description for the menu.
    citations : list[str]
        BibTeX keys (verified against
        docs/dev/synthesizer_parity_citations.md in tests).
    status : str, optional
        "stable" (default), "experimental", or "deprecated".

    Returns
    -------
    Callable
        Decorator for the factory function.

    Examples
    --------
    >>> @register_preset(
    ...     "my_model",
    ...     short_doc="Custom preset description",
    ...     citations=["Author_2024", "Other_2023"],
    ...     status="experimental",
    ... )
    ... def my_preset(redshift=1.0):
    ...     # build and return (model, params)
    ...     pass
    """

    def _decorator(fn: Callable) -> Callable:
        _PRESET_REGISTRY[name] = PresetEntry(
            name=name,
            callable=fn,
            short_doc=short_doc,
            citations=citations,
            status=status,
        )
        return fn

    return _decorator


def list_presets() -> _RegistryTable:
    """List all registered presets.

    Returns
    -------
    _RegistryTable
        One row per preset, with columns ``name``, ``short_doc``,
        ``citations`` and ``status``.

    Notes
    -----
    Returned ``dict[str, dict]`` before #1574; every discovery verb
    returns a table (#1285). ``.names()`` replaces ``list(presets)``, and
    ``{row["name"]: row for row in list_presets()}`` reproduces the old
    name-to-metadata mapping. For one preset, prefer
    :func:`describe_preset`.

    Examples
    --------
    >>> list_presets().names()
    >>> list_presets().filter(status="production")
    """
    return _RegistryTable(
        [
            {
                "name": entry.name,
                "kind": "preset",
                "short_doc": entry.short_doc,
                "citations": entry.citations,
                "status": entry.status,
                "use": f"tengri.presets.describe_preset({entry.name!r})",
            }
            for entry in _PRESET_REGISTRY.values()
        ]
    )


def describe_preset(name: str) -> dict:
    """Get full metadata for a preset.

    Parameters
    ----------
    name : str
        Preset name.

    Returns
    -------
    dict
        Metadata with keys: name, short_doc, citations, status,
        description (full docstring from factory).

    Raises
    ------
    KeyError
        If preset does not exist.

    Examples
    --------
    >>> desc = describe_preset("synthesizer_default")
    >>> print(desc["citations"])
    ['Bruzual_2003', 'Calzetti_2000', ...]
    """
    if name not in _PRESET_REGISTRY:
        available = list(_PRESET_REGISTRY.keys())
        raise KeyError(f"Unknown preset '{name}'. Available: {available}.")

    entry = _PRESET_REGISTRY[name]
    return {
        "name": entry.name,
        "short_doc": entry.short_doc,
        "citations": entry.citations,
        "status": entry.status,
        "description": (entry.callable.__doc__ or "").strip(),
    }


__all__ = [
    "PresetEntry",
    "describe_preset",
    "list_presets",
    "register_preset",
]
