# SPDX-License-Identifier: BSD-3-Clause
"""Runtime registry of IGM transmission models.

Same shape as ``tengri.components.xray._models`` /
``tengri.components.radio._models``. ``_VALID_IGM_TYPES`` derives
from the registry keys per ADR-0005 / ADR-0008.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IGMRegistryEntry:
    """Registry entry for an IGM transmission model."""

    callable: Callable | None
    citation: str = ""
    status: str = "production"
    short_doc: str = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        fn = object.__getattribute__(self, "callable")
        if fn is None:
            raise TypeError("The 'none' IGM model has no callable.")
        return fn(*args, **kwargs)


IGM_MODELS: dict[str, IGMRegistryEntry] = {}


def register_igm_model(
    name: str,
    *,
    citation: str = "",
    status: str = "production",
    short_doc: str = "",
) -> Callable[[Callable | None], Callable | None]:
    """Register an IGM transmission model in :data:`IGM_MODELS`."""

    def decorator(fn: Callable | None) -> Callable | None:
        IGM_MODELS[name] = IGMRegistryEntry(
            callable=fn, citation=citation, status=status, short_doc=short_doc
        )
        return fn

    return decorator
