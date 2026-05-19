# SPDX-License-Identifier: BSD-3-Clause
r"""Deprecation shim: ``tengri.core`` was renamed to ``tengri.protocols``.

The old name was misleading — the package holds *protocol* definitions
(``SEDComponent``, ``DerivedBundle``, ``ForwardState``, etc.), not "core
business logic". The forward-model orchestrator lives in
``tengri.forward.sed_model``.

This shim re-exports everything from ``tengri.protocols`` and emits a
one-shot ``DeprecationWarning`` on first import. Will be removed in
tengri v1.0.

Migration: ``s/tengri\.core/tengri.protocols/g`` across imports.
"""

from __future__ import annotations

import importlib as _importlib
import sys as _sys
import warnings as _warnings

_warnings.warn(
    "tengri.core has been renamed to tengri.protocols and will be removed "
    "in tengri v1.0. Update imports: `from tengri.core ...` → `from "
    "tengri.protocols ...`.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-route every dotted submodule access (``tengri.core.component`` →
# ``tengri.protocols.component``) by aliasing sys.modules entries.
_canonical = _importlib.import_module("tengri.protocols")
_sys.modules[__name__] = _canonical
for _name in ("component", "derived_bundle", "likelihood", "observation"):
    _sub = _importlib.import_module(f"tengri.protocols.{_name}")
    _sys.modules[f"{__name__}.{_name}"] = _sub
