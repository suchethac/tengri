"""Inference backend registry — single source of truth for fitter.run dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BackendEntry:
    """Registry entry for an inference backend."""

    name: str
    runner: Callable
    tier: str = "experimental"  # "primary" | "experimental"
    short_doc: str = ""
    requires: tuple[str, ...] = field(default_factory=tuple)  # optional dep names


_BACKENDS: dict[str, BackendEntry] = {}


def register_backend(
    name: str,
    *,
    tier: str = "experimental",
    short_doc: str = "",
    requires: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
):
    """Decorator to register an inference backend.

    Parameters
    ----------
    name : str
        Canonical method name (e.g., "map", "mcmc_nuts").
    tier : str
        "primary" for primary methods, "experimental" otherwise.
    short_doc : str
        Brief description of the method.
    requires : tuple[str, ...]
        Optional dependency import names (e.g., ("blackjax",)).
    aliases : tuple[str, ...]
        Additional names that map to this backend.
    """

    def deco(fn):
        entry = BackendEntry(
            name=name, runner=fn, tier=tier, short_doc=short_doc, requires=requires
        )
        _BACKENDS[name] = entry
        for a in aliases:
            _BACKENDS[a] = entry
        return fn

    return deco


def get_backend(name: str) -> BackendEntry:
    """Retrieve a backend by name.

    Parameters
    ----------
    name : str
        Method name.

    Returns
    -------
    BackendEntry
        The backend registry entry.

    Raises
    ------
    ValueError
        If the method is not registered.
    """
    if name not in _BACKENDS:
        available = sorted(set(_BACKENDS))
        raise ValueError(f"Unknown inference method '{name}'. Available: {available}")
    return _BACKENDS[name]


def all_backends() -> list[BackendEntry]:
    """Return all registered backends, deduplicated and sorted.

    Returns
    -------
    list[BackendEntry]
        Backends sorted by (tier != "primary", name).
    """
    seen, out = set(), []
    for entry in _BACKENDS.values():
        if id(entry) in seen:
            continue
        seen.add(id(entry))
        out.append(entry)
    return sorted(out, key=lambda e: (e.tier != "primary", e.name))
