# SPDX-License-Identifier: BSD-3-Clause
"""Deprecated shim — module renamed to :mod:`tengri.protocols.derived_state`.

Imports of ``tengri.protocols.derived_bundle`` continue to work but
will be removed in tengri v1.0.
"""

from __future__ import annotations

import warnings as _warnings

from tengri.protocols.derived_state import DerivedState

_warnings.warn(
    "tengri.protocols.derived_bundle is deprecated; use tengri.protocols.derived_state instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Old name kept for one release.
DerivedBundle = DerivedState

__all__ = ["DerivedBundle"]
