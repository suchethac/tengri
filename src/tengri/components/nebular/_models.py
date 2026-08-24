# SPDX-License-Identifier: BSD-3-Clause
"""Runtime registry of nebular emission backends.

Mirrors ``tengri.components.xray._models``, ``tengri.components.radio._models``,
and ``tengri.components.igm._models`` (#355) and the older
``SFH_REGISTRY`` / ``AGN_MODELS`` patterns. ``_VALID_NEBULAR_TYPES``
in :mod:`tengri.parameters.groups` derives from
:data:`NEBULAR_MODELS.keys()` per ADR-0005 / ADR-0008 (single source of
truth): adding a new backend is one ``register_nebular_model`` call,
not a parallel edit to a validator set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NebularRegistryEntry:
    """Registry entry for a nebular emission backend.

    Attributes
    ----------
    callable: Callable or None
        The backend class (or any reference). ``None`` for the
        ``'none'`` disable-toggle. Dispatch in
        ``tengri.parameters.groups._translate_neb`` still routes
        through ``nebular_ssp`` / ``nebular_cue`` / ``nebular`` flags
        on :class:`Parameters`; this field is metadata.
    citation: str
        Academic citation. Empty for the disable-toggle.
    status: str
        ``"production"`` / ``"experimental"`` / ``"demo"`` / ``"deprecated"``.
    short_doc: str
        One-line description.
    """

    callable: Callable | None
    citation: str = ""
    status: str = "production"
    short_doc: str = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        fn = object.__getattribute__(self, "callable")
        if fn is None:
            raise TypeError("The 'none' nebular model has no callable.")
        return fn(*args, **kwargs)


NEBULAR_MODELS: dict[str, NebularRegistryEntry] = {}


def register_nebular_model(
    name: str,
    *,
    citation: str = "",
    status: str = "production",
    short_doc: str = "",
) -> Callable[[Callable | None], Callable | None]:
    """Register a nebular emission backend in :data:`NEBULAR_MODELS`."""

    def decorator(fn: Callable | None) -> Callable | None:
        NEBULAR_MODELS[name] = NebularRegistryEntry(
            callable=fn, citation=citation, status=status, short_doc=short_doc
        )
        return fn

    return decorator
