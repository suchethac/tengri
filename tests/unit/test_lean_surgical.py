"""Unit tests for surgical lean clearing (Phase B).

Tests verify that:
- lean=True drops only inference-body caches (scope="inference_body")
- Forward function caches survive between runs
- tengri.gc() clears all caches including structural
"""

from __future__ import annotations

import pytest

import tengri
from tengri.inference._model_cache import (
    _STRUCTURAL_KERNEL_CACHE,
    get_structural_kernel_cache,
)
from tengri.inference.jit_engine import (
    _SHARED_ENGINE_CACHE,
    _SHARED_GRAD_FN_CACHE,
    _SHARED_LOGDENSITY_FN_CACHE,
    _SHARED_LOSS_FN_CACHE,
    clear_shared_caches,
)


@pytest.mark.unit
def test_clear_shared_caches_inference_body_scope():
    """scope='inference_body' drops only engine and per-model caches."""
    # Populate all caches with dummy entries
    _SHARED_ENGINE_CACHE[("engine_key",)] = "engine_value"
    _SHARED_LOSS_FN_CACHE[("loss_key",)] = "loss_value"
    _SHARED_GRAD_FN_CACHE[("grad_key",)] = "grad_value"
    _SHARED_LOGDENSITY_FN_CACHE[("logdensity_key",)] = "logdensity_value"

    sig = ("test_struct_sig",)
    struct_cache = get_structural_kernel_cache(sig)
    struct_cache["kernel"] = "kernel_value"

    # Clear with inference_body scope
    clear_shared_caches(scope="inference_body", drop_xla=False)

    # Engine cache should be cleared
    assert len(_SHARED_ENGINE_CACHE) == 0, "Engine cache should be cleared"

    # Forward function caches should survive
    assert len(_SHARED_LOSS_FN_CACHE) > 0, "Loss cache should survive inference_body clear"
    assert len(_SHARED_GRAD_FN_CACHE) > 0, "Grad cache should survive inference_body clear"
    assert len(_SHARED_LOGDENSITY_FN_CACHE) > 0, (
        "Logdensity cache should survive inference_body clear"
    )

    # Structural cache should survive
    assert len(_STRUCTURAL_KERNEL_CACHE) > 0, (
        "Structural cache should survive inference_body clear"
    )


@pytest.mark.unit
def test_clear_shared_caches_all_scope():
    """scope='all' (default) drops everything."""
    # Populate all caches
    _SHARED_ENGINE_CACHE[("engine_key",)] = "engine_value"
    _SHARED_LOSS_FN_CACHE[("loss_key",)] = "loss_value"
    _SHARED_GRAD_FN_CACHE[("grad_key",)] = "grad_value"
    _SHARED_LOGDENSITY_FN_CACHE[("logdensity_key",)] = "logdensity_value"

    sig = ("test_struct_sig",)
    struct_cache = get_structural_kernel_cache(sig)
    struct_cache["kernel"] = "kernel_value"

    # Clear with all scope
    clear_shared_caches(scope="all", drop_xla=False)

    # All caches should be empty
    assert len(_SHARED_ENGINE_CACHE) == 0, "Engine cache should be cleared"
    assert len(_SHARED_LOSS_FN_CACHE) == 0, "Loss cache should be cleared"
    assert len(_SHARED_GRAD_FN_CACHE) == 0, "Grad cache should be cleared"
    assert len(_SHARED_LOGDENSITY_FN_CACHE) == 0, "Logdensity cache should be cleared"
    assert len(_STRUCTURAL_KERNEL_CACHE) == 0, "Structural cache should be cleared"


@pytest.mark.unit
def test_clear_shared_caches_invalid_scope():
    """Invalid scope raises ValueError."""
    with pytest.raises(ValueError, match="scope must be 'all' or 'inference_body'"):
        clear_shared_caches(scope="invalid")


@pytest.mark.unit
def test_lean_context_manager():
    """lean context manager sets lean mode flag."""
    import tengri.inference.jit_engine as jit_engine

    assert not jit_engine._LEAN_MODE, "Lean mode should start false"

    with tengri.lean():
        assert jit_engine._LEAN_MODE, "Lean mode should be true inside context"

    assert not jit_engine._LEAN_MODE, "Lean mode should be restored after context"


@pytest.mark.unit
def test_clear_shared_caches_keep_sig_drops_only_stale():
    """keep_sig preserves the matching entry, drops others (smart lean)."""
    sig_a = (("model_sig_a",), "mcmc_hmc")
    sig_b = (("model_sig_a",), "map")
    sig_c = (("model_sig_other",), "mcmc_hmc")

    _SHARED_ENGINE_CACHE.clear()
    _SHARED_ENGINE_CACHE[sig_a] = "hmc_compile"
    _SHARED_ENGINE_CACHE[sig_b] = "map_compile"
    _SHARED_ENGINE_CACHE[sig_c] = "other_compile"

    # Lean about to run (model_sig_a, mcmc_hmc): should keep sig_a, drop sig_b/c.
    clear_shared_caches(scope="inference_body", drop_xla=False, keep_sig=sig_a)

    assert sig_a in _SHARED_ENGINE_CACHE, "matching entry must be kept"
    assert sig_b not in _SHARED_ENGINE_CACHE, "stale prior-phase entry must be dropped"
    assert sig_c not in _SHARED_ENGINE_CACHE, "different-shape entry must be dropped"
    assert _SHARED_ENGINE_CACHE[sig_a] == "hmc_compile", "kept entry must be intact"


@pytest.mark.unit
def test_clear_shared_caches_keep_sig_no_match_drops_all():
    """keep_sig that matches nothing in cache drops everything in scope."""
    _SHARED_ENGINE_CACHE.clear()
    _SHARED_ENGINE_CACHE[(("a",), "map")] = "x"
    _SHARED_ENGINE_CACHE[(("b",), "map")] = "y"

    clear_shared_caches(scope="inference_body", drop_xla=False, keep_sig=(("nonexistent",), "map"))

    assert len(_SHARED_ENGINE_CACHE) == 0


@pytest.mark.unit
def test_gc_calls_clear_all():
    """tengri.gc() clears all caches including structural."""
    # Populate caches
    _SHARED_ENGINE_CACHE[("key",)] = "value"
    _SHARED_LOSS_FN_CACHE[("key",)] = "value"
    sig = ("sig",)
    get_structural_kernel_cache(sig)["key"] = "value"

    # Call gc
    tengri.gc()

    # All should be empty
    assert len(_SHARED_ENGINE_CACHE) == 0
    assert len(_SHARED_LOSS_FN_CACHE) == 0
    assert len(_STRUCTURAL_KERNEL_CACHE) == 0
