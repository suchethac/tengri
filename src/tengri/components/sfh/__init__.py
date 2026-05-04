# SPDX-License-Identifier: BSD-3-Clause
"""Deprecation shim for ``tengri.components.sfh``.

This package was relocated to :mod:`tengri.components.stellar.sfh` in
Phase II-2.1 of the SEDComponent migration. This shim keeps existing
imports working with a one-shot :class:`DeprecationWarning`. The shim
will be removed in tengri v1.0.

Notes
-----
The shim eagerly imports :mod:`tengri.components.stellar.sfh` (which
loads the package and every submodule via :func:`importlib.import_module`)
and then aliases each submodule under the old dotted path via
:data:`sys.modules`. This means ``from tengri.components.sfh.gp_sfh
import gp_from_xi`` continues to resolve without stub files on disk.
"""

from __future__ import annotations

import importlib as _importlib
import sys
import warnings

import tengri.components.stellar.sfh as _new

warnings.warn(
    "tengri.components.sfh has been relocated to "
    "tengri.components.stellar.sfh and will be removed in tengri v1.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = list(getattr(_new, "__all__", []))
for _name in __all__:
    globals()[_name] = getattr(_new, _name)

_SUBMODULES = (
    "chemical_evolution",
    "dense_basis",
    "gp_sfh",
    "mean_sfh",
    "met_registry",
    "metallicity_history",
    "nonparametric",
    "psd_models",
    "registry",
)
for _sub in _SUBMODULES:
    _new_dotted = f"tengri.components.stellar.sfh.{_sub}"
    _old_dotted = f"tengri.components.sfh.{_sub}"
    _importlib.import_module(_new_dotted)
    sys.modules[_old_dotted] = sys.modules[_new_dotted]

del sys, warnings, _importlib, _new, _name, _sub, _new_dotted, _old_dotted
del _SUBMODULES
