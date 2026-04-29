"""Per-model runtime cache using a WeakKeyDictionary.

Replaces the pattern of monkey-patching private dicts onto SEDModel at runtime
(model._jit_engine_cache = {}, model._loss_fn_cache = {}, etc.).

Keyed on the SEDModel object identity. When ``precompute_spectroscopy`` /
``precompute_ztable`` later return a new SEDModel (Phase 4), the new instance
gets a fresh cache and the JIT functions recompile on first use — an
acceptable cost since the cached artefacts can always be regenerated.
"""

from __future__ import annotations

import weakref
from typing import Any

_caches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def get_model_cache(model: Any) -> dict:
    """Return the shared mutable namespace dict for *model*."""
    try:
        return _caches[model]
    except KeyError:
        _caches[model] = {}
        return _caches[model]


def clear_model_cache(model: Any) -> None:
    """Drop *model*'s cached compiled functions (loss fns, JIT engines, etc.).

    Use after a state mutation that invalidates compiled artefacts
    (``precompute_spectroscopy``, ``precompute_ztable``). The cache is
    repopulated lazily on next inference call. The cached artefacts are
    JIT-compiled functions; recomputing them is a one-time first-call cost.
    """
    _caches.pop(model, None)
