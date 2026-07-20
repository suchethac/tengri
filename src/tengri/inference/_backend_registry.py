# SPDX-License-Identifier: BSD-3-Clause
"""Inference backend registry — single source of truth for fitter.run dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BackendEntry:
    """Registry entry for an inference backend.

    Parameters
    ----------
    name : str
        Canonical method name (e.g., ``"map"``, ``"mcmc_nuts"``).
    runner : Callable
        Backend entry point. Signature depends on ``legacy_fitter``:

        - ``legacy_fitter=True``  → ``runner(fitter, *, key, **kwargs)``
        - ``legacy_fitter=False`` → ``runner(context, *, key, **kwargs)``

        where ``context`` is an :class:`InferenceContext`.
    tier : str
        ``"primary"`` for promoted methods, ``"experimental"`` otherwise.
    short_doc : str
        Brief description.
    requires : tuple[str, ...]
        Optional dependency import names (e.g. ``("blackjax",)``).
    legacy_fitter : bool
        If ``True`` (default), ``Fitter.run`` passes the full Fitter to
        the runner. Set to ``False`` for backends migrated to the
        :class:`InferenceContext` Protocol. The flag is removed once all
        backends migrate (see ADR-0010 / final PR of the inference-
        backend refactor).
    """

    name: str
    runner: Callable
    tier: str = "experimental"  # "primary" | "experimental"
    short_doc: str = ""
    requires: tuple[str, ...] = field(default_factory=tuple)  # optional dep names
    legacy_fitter: bool = True
    # Predicate called with whatever ``runner`` receives (Fitter or InferenceContext).
    # Returns True if this backend can run for the given target's spec/dims/dtypes.
    # Default ``None`` means "no compatibility constraint" (always usable).
    is_compatible: Callable[[Any], bool] | None = None


_BACKENDS: dict[str, BackendEntry] = {}


def register_backend(
    name: str,
    *,
    tier: str = "experimental",
    short_doc: str = "",
    requires: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
    legacy_fitter: bool = True,
    is_compatible: Callable[[Any], bool] | None = None,
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
            name=name,
            runner=fn,
            tier=tier,
            short_doc=short_doc,
            requires=requires,
            legacy_fitter=legacy_fitter,
            is_compatible=is_compatible,
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
        # Group available methods by tier so the error is digestible at
        # 19 backends. Suggest the discovery API rather than dumping a flat
        # list of every experimental sampler.
        primary = sorted({e.name for e in _BACKENDS.values() if e.tier == "primary"})
        raise ValueError(
            f"Unknown inference method '{name}'.  "
            f"Recommended (tier=primary): {primary}.  "
            "Run `tengri.list_inference_methods()` for the full list including "
            "experimental backends."
        )
    return _BACKENDS[name]


def check_requires(entry: BackendEntry) -> None:
    """Verify the backend's optional dependencies are importable.

    Raises a friendly ImportError before the runner crashes deep in a
    third-party package. Called by ``Fitter.run`` just before dispatch.

    Parameters
    ----------
    entry : BackendEntry
        The backend whose ``requires`` tuple should be checked.

    Raises
    ------
    ImportError
        If any required dependency cannot be imported. The error message
        names the offending package and gives the recommended pip extra.
    """
    import importlib

    _PIP_EXTRA: dict[str, str] = {
        "blackjax": 'pip install "tengri[blackjax]"  (or:  pip install blackjax)',
        "nifty8": "pip install nifty8.re",
        "optax": "pip install optax",
        "jaxopt": "pip install jaxopt",
        "dynesty": "pip install dynesty",
    }
    for pkg in entry.requires:
        try:
            importlib.import_module(pkg)
        except ImportError as exc:
            hint = _PIP_EXTRA.get(pkg, f"pip install {pkg}")
            raise ImportError(
                f"Inference method '{entry.name}' requires {pkg!r}, "
                f"which is not installed.  Install it with:\n    {hint}"
            ) from exc


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
