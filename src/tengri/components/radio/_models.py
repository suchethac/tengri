# SPDX-License-Identifier: BSD-3-Clause
"""Runtime registry of radio emission models.

Same shape as ``tengri.components.xray._models``. ``_VALID_RADIO_TYPES``
derives from the registry keys per ADR-0005 / ADR-0008.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RadioRegistryEntry:
    """Registry entry for a radio emission model."""

    callable: Callable | None
    citation: str = ""
    status: str = "production"
    short_doc: str = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        fn = object.__getattribute__(self, "callable")
        if fn is None:
            raise TypeError("The 'none' radio model has no callable.")
        return fn(*args, **kwargs)


RADIO_MODELS: dict[str, RadioRegistryEntry] = {}


def register_radio_model(
    name: str,
    *,
    citation: str = "",
    status: str = "production",
    short_doc: str = "",
) -> Callable[[Callable | None], Callable | None]:
    """Register a radio emission model in :data:`RADIO_MODELS`."""

    def decorator(fn: Callable | None) -> Callable | None:
        RADIO_MODELS[name] = RadioRegistryEntry(
            callable=fn, citation=citation, status=status, short_doc=short_doc
        )
        return fn

    return decorator
