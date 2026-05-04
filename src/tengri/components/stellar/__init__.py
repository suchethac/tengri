# SPDX-License-Identifier: BSD-3-Clause
"""Stellar SEDComponent — skeleton for Phase II-2 migration.

The body of the migration (merging :mod:`tengri.components.stellar.sfh` and
:mod:`tengri.components.stellar.sps` into a unified ``StellarSEDComponent``)
is deferred. This package currently exports only the configured
**skeleton** so downstream adapters can be designed against a stable
contract.

See ``docs/dev/phase_ii_2_stellar_migration.md`` for the migration plan
and design decisions (resolved 2026-05-03).
"""

from __future__ import annotations

from tengri.components.stellar.component import (
    StellarSEDComponent,
    StellarSEDComponentConfig,
    StellarSEDComponentState,
)

__all__ = [
    "StellarSEDComponent",
    "StellarSEDComponentConfig",
    "StellarSEDComponentState",
]
