# SPDX-License-Identifier: BSD-3-Clause
"""Deprecation shim for ``tengri.components.sps``.

This package was relocated to :mod:`tengri.components.stellar.sps` in
Phase II-2.1 of the SEDComponent migration. This shim keeps existing
imports working with a one-shot :class:`DeprecationWarning`. The shim
will be removed in tengri v1.0.

Notes
-----
The shim eagerly imports :mod:`tengri.components.stellar.sps` (loading
each submodule via :func:`importlib.import_module`) and aliases each
under the old dotted path via :data:`sys.modules`. This means ``from
tengri.components.sps.dsps_wrapper import SSPData`` continues to resolve
without stub files on disk.
"""

from __future__ import annotations

import importlib as _importlib
import sys
import warnings

import tengri.components.stellar.sps as _new

warnings.warn(
    "tengri.components.sps has been relocated to "
    "tengri.components.stellar.sps and will be removed in tengri v1.0.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the public surface that the old __init__.py advertised.
_REEXPORTS = (
    "SSPData",
    "compute_csp_weights",
    "effective_metallicity",
    "interpolate_met_alpha",
    "load_ssp_data",
)
for _name in _REEXPORTS:
    if hasattr(_new, _name):
        globals()[_name] = getattr(_new, _name)

__all__ = [_n for _n in _REEXPORTS if _n in globals()]

_SUBMODULES = (
    "dsps_wrapper",
    "mass_remaining",
    "precompute",
)
for _sub in _SUBMODULES:
    _new_dotted = f"tengri.components.stellar.sps.{_sub}"
    _old_dotted = f"tengri.components.sps.{_sub}"
    _importlib.import_module(_new_dotted)
    sys.modules[_old_dotted] = sys.modules[_new_dotted]

del sys, warnings, _importlib, _new, _name, _sub, _new_dotted, _old_dotted
del _REEXPORTS, _SUBMODULES
