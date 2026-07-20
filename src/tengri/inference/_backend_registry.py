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
        ``"primary"`` for promoted methods, ``"experimental"`` for backends
        that work but are not yet validated, ``"broken"`` for backends known
        to produce wrong answers or crash. ``"broken"`` is hidden from the
        default :func:`tengri.list_inference_methods` listing and refused by
        ``Fitter.run`` unless the caller passes ``allow_unvalidated=True``.
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


#: The tiers a backend may declare.
#:
#: ``"broken"`` is not a softer ``"experimental"``. Experimental means "works,
#: not yet validated"; broken means the backend's own ``short_doc`` says it
#: returns wrong answers (``[POOR MIXING]``) or crashes the process
#: (``[UNSTABLE]``). Five backends carried such a warning while sitting in the
#: experimental tier, indistinguishable from ones that work (#1287).
TIERS: frozenset[str] = frozenset({"primary", "experimental", "broken"})

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
        One of :data:`TIERS`. ``"primary"`` for promoted methods,
        ``"experimental"`` for working-but-unvalidated ones, ``"broken"``
        for backends known to return wrong answers or crash.
    short_doc : str
        Brief description of the method.
    requires : tuple[str, ...]
        Optional dependency import names (e.g., ("blackjax",)).
    aliases : tuple[str, ...]
        Additional names that map to this backend.

    Raises
    ------
    ValueError
        If ``tier`` is not a recognized tier. A typo would otherwise create a
        silent third tier that no filter matches.
    """
    if tier not in TIERS:
        raise ValueError(
            f"register_backend({name!r}): unknown tier {tier!r}. Valid tiers: {sorted(TIERS)}."
        )

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


def check_usable(entry: BackendEntry, *, allow_unvalidated: bool = False) -> None:
    """Refuse to run a backend that is known to give wrong answers (#1287).

    Five backends declared ``[POOR MIXING]`` or ``[UNSTABLE]`` in their own
    ``short_doc`` while sitting at ``tier="experimental"`` — the same tier as
    backends that work. A user who picked ``mcmc_ghmc`` because it is "fast
    (cold ~17s)" got R-hat ~ 2.5-3.1 and no runtime signal that the chains
    had not converged.

    Wrongness that only a doc string mentions is wrongness that ships. This
    makes the caller say out loud that they accept it.

    Parameters
    ----------
    entry : BackendEntry
        The backend about to be dispatched.
    allow_unvalidated : bool, optional
        Escape hatch for benchmarking and backend development. Default False.

    Raises
    ------
    BackendError
        If ``entry.tier == "broken"`` and ``allow_unvalidated`` is False. The
        message carries the backend's own diagnosis verbatim.
    """
    if entry.tier != "broken" or allow_unvalidated:
        return

    from tengri.config.exceptions import BackendError

    primary = sorted({e.name for e in _BACKENDS.values() if e.tier == "primary"})
    raise BackendError(
        f"Inference method '{entry.name}' is registered as tier='broken' and "
        f"will not run by default.\n\n"
        f"  {entry.short_doc}\n\n"
        f"Working alternatives (tier=primary): {primary}.\n"
        f"To run it anyway -- for benchmarking or backend development, not for "
        f"science -- pass allow_unvalidated=True."
    )


def all_backends(*, include_broken: bool = True) -> list[BackendEntry]:
    """Return all registered backends, deduplicated and sorted.

    Parameters
    ----------
    include_broken : bool, optional
        Include ``tier="broken"`` entries. Default True, so internal callers
        that need the complete registry (dispatch, conformance tests) keep
        seeing everything; the user-facing listing opts out.

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
        if not include_broken and entry.tier == "broken":
            continue
        out.append(entry)
    return sorted(out, key=lambda e: (e.tier != "primary", e.name))
