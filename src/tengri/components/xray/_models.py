# SPDX-License-Identifier: BSD-3-Clause
"""Runtime registry of X-ray emission models.

Mirrors ``SFH_REGISTRY`` (``components/stellar/sfh/registry.py``) and
``AGN_MODELS`` (``components/agn/unified.py``): a flat dict keyed by
variant name with metadata fields that :func:`tengri.list_xray_models`
introspects. Adding a new X-ray model = one ``register_xray_model``
call below; ``_VALID_XRAY_TYPES`` derives from this registry per
ADR-0005 / ADR-0008 (single source of truth).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class XRayRegistryEntry:
    """Registry entry for an X-ray emission model.

    Attributes
    ----------
    callable: Callable or None
        The X-ray model function. ``None`` for the ``'none'`` toggle.
    citation: str
        Academic citation. Empty string for the disable-toggle.
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
            raise TypeError("The 'none' X-ray model has no callable.")
        return fn(*args, **kwargs)


XRAY_MODELS: dict[str, XRayRegistryEntry] = {}


def register_xray_model(
    name: str,
    *,
    citation: str = "",
    status: str = "production",
    short_doc: str = "",
) -> Callable[[Callable | None], Callable | None]:
    """Register an X-ray emission model in :data:`XRAY_MODELS`.

    Can be used either as a decorator on the model callable or invoked
    directly with ``callable=None`` for the disable-toggle entry::

        @register_xray_model("simple", citation=..., short_doc=...)
        def xray_total(...): ...

        register_xray_model("none", short_doc="Disable")(None)
    """

    def decorator(fn: Callable | None) -> Callable | None:
        XRAY_MODELS[name] = XRayRegistryEntry(
            callable=fn, citation=citation, status=status, short_doc=short_doc
        )
        return fn

    return decorator
