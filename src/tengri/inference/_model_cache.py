"""Per-model runtime cache using a WeakKeyDictionary.

Replaces the pattern of monkey-patching private dicts onto SEDModel at runtime
(model._jit_engine_cache = {}, model._loss_fn_cache = {}, etc.).
Caches are GC'd automatically when the model is.
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
