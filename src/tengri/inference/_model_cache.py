# SPDX-License-Identifier: BSD-3-Clause
"""Per-model runtime cache using a WeakKeyDictionary.

Replaces the pattern of monkey-patching private dicts onto SEDModel at runtime
(model._jit_engine_cache = {}, model._loss_fn_cache = {}, etc.).

Keyed on the SEDModel object identity. When ``precompute_spectroscopy`` /
``precompute_ztable`` later return a new SEDModel, the new instance
gets a fresh cache and the JIT functions recompile on first use, an
acceptable cost since the cached artefacts can always be regenerated.

Structural kernel cache: two SEDModel instances with identical compile_signature()
share their prediction-kernel JIT compiles via
``ModelCacheOwner.get_structural_kernel()``.

All state lives on ``ModelCacheOwner`` instances; the module-level
``_default_owner`` singleton backs the deprecated free functions below and
the internal callers in ``fitter.py`` / ``jit_engine.py`` / ``backends/``.
"""

from __future__ import annotations

import os
import threading
import warnings
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelCacheOwner:
    """Explicit owner of per-model and structural kernel cache state.

    Manages a per-SEDModel WeakKeyDictionary cache and a bounded LRU cache
    for prediction kernels shared across models with identical compile signatures.
    Mirrors the design of CompileCache in jit_engine.py.

    Parameters
    ----------
    max_kernel_entries : int, optional
        Maximum number of structural kernel cache entries to hold before
        evicting oldest. Default 4; tune via TENGRI_STRUCTURAL_CACHE_MAXSIZE.
    _model_caches : weakref.WeakKeyDictionary
        Per-model runtime namespaces keyed on SEDModel object identity.
        Do not access directly; use get_or_compile_model().
    _kernel_cache : OrderedDict
        Structural kernel cache for JIT functions keyed on compile signature.
        Do not access directly; use get_structural_kernel().
    _lock : threading.Lock
        Thread safety for concurrent access.

    Attributes
    ----------
    max_kernel_entries : int
    """

    max_kernel_entries: int = field(
        default_factory=lambda: int(os.environ.get("TENGRI_STRUCTURAL_CACHE_MAXSIZE", "4"))
    )
    _model_caches: weakref.WeakKeyDictionary = field(default_factory=weakref.WeakKeyDictionary)
    _kernel_cache: OrderedDict[Any, Any] = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get_or_compile_model(self, model: Any, build_fn=None) -> dict:
        """Get or create the shared namespace dict for *model*.

        Parameters
        ----------
        model : SEDModel
            Model instance (key identity).
        build_fn : callable, optional
            Unused; present for API symmetry with CompileCache.get_or_compile().
            The model cache is always created empty and populated lazily by consumers.

        Returns
        -------
        dict
            The mutable namespace dict for this model.
        """
        with self._lock:
            try:
                return self._model_caches[model]
            except KeyError:
                self._model_caches[model] = {}
                return self._model_caches[model]

    def get_structural_kernel(self, signature: tuple, build_fn=None) -> dict:
        """Get or create the shared kernel namespace for this signature.

        Two SEDModel instances with the same compile_signature() will return
        the same kernel dictionary, enabling sharing of JIT-compiled prediction
        functions. Touching an entry promotes it to MRU; LRU eviction occurs
        at max_kernel_entries.

        Parameters
        ----------
        signature : tuple
            Structural fingerprint from SEDModel.compile_signature().
        build_fn : callable, optional
            Unused; present for API symmetry with CompileCache.get_or_compile().
            The kernel cache is always created empty and populated lazily.

        Returns
        -------
        dict
            Mutable namespace for storing kernel functions keyed on signature.
        """
        with self._lock:
            if signature in self._kernel_cache:
                self._kernel_cache.move_to_end(signature)
                return self._kernel_cache[signature]
            if len(self._kernel_cache) >= self.max_kernel_entries:
                self._kernel_cache.popitem(last=False)
            self._kernel_cache[signature] = {}
            return self._kernel_cache[signature]

    def clear(self) -> None:
        """Clear all cached entries (both model and kernel caches)."""
        with self._lock:
            self._model_caches.clear()
            self._kernel_cache.clear()


# Module-level singleton for backward compatibility.
_default_owner = ModelCacheOwner()


def get_model_cache(model: Any) -> dict:
    """Return the shared mutable namespace dict for *model*.

    .. deprecated::
        Use ModelCacheOwner.get_or_compile_model() instead.
    """
    warnings.warn(
        "get_model_cache() is deprecated; use ModelCacheOwner.get_or_compile_model() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _default_owner.get_or_compile_model(model)


def clear_model_cache(model: Any) -> None:
    """Drop *model*'s cached compiled functions (loss fns, JIT engines, etc.).

    Use after a state mutation that invalidates compiled artefacts
    (``precompute_spectroscopy``, ``precompute_ztable``). The cache is
    repopulated lazily on next inference call. The cached artefacts are
    JIT-compiled functions; recomputing them is a one-time first-call cost.

    .. deprecated::
        Use ModelCacheOwner.get_or_compile_model() then manually pop the model,
        or construct a fresh ModelCacheOwner instance for isolation.
    """
    warnings.warn(
        "clear_model_cache() is deprecated; use a ModelCacheOwner instance instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    with _default_owner._lock:
        _default_owner._model_caches.pop(model, None)


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

    .. deprecated::
        Use ModelCacheOwner.get_structural_kernel() instead.
    """
    warnings.warn(
        "get_structural_kernel_cache() is deprecated; "
        "use ModelCacheOwner.get_structural_kernel() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _default_owner.get_structural_kernel(signature)


def clear_structural_kernel_cache() -> None:
    """Drop all cached prediction kernels.

    Called by clear_shared_caches(scope="all") and tengri.gc().

    .. deprecated::
        Use ModelCacheOwner.clear() instead.
    """
    warnings.warn(
        "clear_structural_kernel_cache() is deprecated; use ModelCacheOwner.clear() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    _default_owner.clear()
