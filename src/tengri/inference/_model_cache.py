"""Per-model runtime cache using a WeakKeyDictionary.

Replaces the pattern of monkey-patching private dicts onto SEDModel at runtime
(model._jit_engine_cache = {}, model._loss_fn_cache = {}, etc.).

Keyed on the SEDModel object identity. When ``precompute_spectroscopy`` /
``precompute_ztable`` later return a new SEDModel (Phase 4), the new instance
gets a fresh cache and the JIT functions recompile on first use — an
acceptable cost since the cached artefacts can always be regenerated.

Structural kernel cache: two SEDModel instances with identical compile_signature()
share their prediction-kernel JIT compiles via _STRUCTURAL_KERNEL_CACHE.
"""

from __future__ import annotations

import os
import threading
import weakref
from collections import OrderedDict
from typing import Any

_caches: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# Structural cache for prediction kernels keyed on SEDModel.compile_signature()
# Enables sharing of JIT'd photometry/spectrum/mock kernels across instances
# with identical structure (same filters, wavelength grid, physics config, etc.)
_STRUCTURAL_KERNEL_CACHE: OrderedDict[tuple, dict] = OrderedDict()
_STRUCTURAL_KERNEL_MAXSIZE = int(os.environ.get("TENGRI_STRUCTURAL_CACHE_MAXSIZE", "4"))
_STRUCTURAL_KERNEL_LOCK = threading.Lock()


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


def get_structural_kernel_cache(signature: tuple) -> dict:
    """Return the shared kernel namespace for any model with this signature.

    Two SEDModel instances with the same compile_signature() will return
    the same kernel dictionary, enabling sharing of JIT-compiled prediction
    functions. Touching an entry promotes it to MRU; LRU eviction occurs
    at maxsize.

    Parameters
    ----------
    signature : tuple
        Structural fingerprint from SEDModel.compile_signature().

    Returns
    -------
    dict
        Mutable namespace for storing kernel functions keyed on signature.
    """
    with _STRUCTURAL_KERNEL_LOCK:
        if signature in _STRUCTURAL_KERNEL_CACHE:
            _STRUCTURAL_KERNEL_CACHE.move_to_end(signature)
            return _STRUCTURAL_KERNEL_CACHE[signature]
        if len(_STRUCTURAL_KERNEL_CACHE) >= _STRUCTURAL_KERNEL_MAXSIZE:
            _STRUCTURAL_KERNEL_CACHE.popitem(last=False)
        _STRUCTURAL_KERNEL_CACHE[signature] = {}
        return _STRUCTURAL_KERNEL_CACHE[signature]


def clear_structural_kernel_cache() -> None:
    """Drop all cached prediction kernels.

    Called by clear_shared_caches(scope="all") and tengri.gc().
    """
    with _STRUCTURAL_KERNEL_LOCK:
        _STRUCTURAL_KERNEL_CACHE.clear()
