# SPDX-License-Identifier: BSD-3-Clause
"""Inference-backend introspection helpers.

Mirrors the role the forward-model kernel-strategy module played (ADR-0004)
on the inference side. That module has since been removed — ``tengri.
KernelStrategy`` is now a tombstone object — so it is named here as history
rather than cross-referenced:

- :class:`BackendStatus` makes the difference between *missing optional
  dep*, *incompatible with the given spec*, and *ready to run* legible —
  the kernel-strategy module's "build failures surface explicitly" rule
  carried over one-to-one.
- :func:`resolve_status` is the canonical predicate used by
  :func:`tengri.registry.list_inference_methods` to populate its
  ``status`` column.

Auto-selection logic (``method="auto"``, ``method="mcmc"``) currently
lives inline in :meth:`Fitter.run`; that path is data-driven enough on
its own that a generic ``BackendStrategy.select(...)`` is deferred until
backends carry richer :attr:`BackendEntry.is_compatible` predicates.
"""

from __future__ import annotations

import importlib
from enum import StrEnum
from typing import Any

from tengri.inference._backend_registry import BackendEntry

__all__ = ["BackendStatus", "resolve_status"]


class BackendStatus(StrEnum):
    """Whether a backend can run right now.

    Members
    -------
    OK
        Importable and compatible with the supplied target (or no target
        was supplied — compatibility deferred to run time).
    MISSING_DEP
        At least one entry in :attr:`BackendEntry.requires` is not
        importable. ``pip install <pkg>`` fixes it.
    INCOMPATIBLE
        Optional deps are installed but the backend's
        :attr:`BackendEntry.is_compatible` predicate rejected the target
        (e.g. dimensionality too high for NUTS).
    """

    OK = "ok"
    MISSING_DEP = "missing_dep"
    INCOMPATIBLE = "incompatible"


def _check_deps_importable(entry: BackendEntry) -> bool:
    for pkg in entry.requires:
        try:
            importlib.import_module(pkg)
        except ImportError:
            return False
    return True


def resolve_status(entry: BackendEntry, target: Any | None = None) -> BackendStatus:
    """Classify a backend's readiness to run.

    Parameters
    ----------
    entry : BackendEntry
        Registry entry for the backend.
    target : Fitter | InferenceContext, optional
        If supplied, evaluated against ``entry.is_compatible``. If
        ``None`` (or the entry has no predicate), compatibility is
        deferred to dispatch time and only dependency presence is
        checked.

    Returns
    -------
    BackendStatus
        See :class:`BackendStatus`.
    """
    if not _check_deps_importable(entry):
        return BackendStatus.MISSING_DEP
    if target is not None and entry.is_compatible is not None:
        try:
            if not entry.is_compatible(target):
                return BackendStatus.INCOMPATIBLE
        except Exception:  # pragma: no cover - predicate bugs shouldn't crash listing
            return BackendStatus.INCOMPATIBLE
    return BackendStatus.OK
