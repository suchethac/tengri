"""Unit tests for structural kernel caching (Phase A).

Tests verify that:
- Two SEDModel instances with identical compile_signature() share prediction kernels
- Kernels are not shared when signatures differ
- LRU eviction works correctly
- Clear helpers propagate through tengri.gc()
"""

from __future__ import annotations

import pytest

import tengri
from tengri.inference._model_cache import (
    _default_owner,
    clear_structural_kernel_cache,
    get_structural_kernel_cache,
)

pytestmark = pytest.mark.contract

# Reach the real structural cache via the singleton ``ModelCacheOwner``;
# the module-level ``_STRUCTURAL_KERNEL_CACHE`` global is a stale legacy
# alias that no longer reflects current state after the cache-owner
# refactor.
_STRUCTURAL_KERNEL_CACHE = _default_owner._kernel_cache
_STRUCTURAL_KERNEL_MAXSIZE = _default_owner.max_kernel_entries


@pytest.fixture(autouse=True)
def _isolate_structural_cache():
    """Reset the structural cache around each test.

    The cache is a process-wide singleton. Other tests in the suite leave
    real SEDModel signatures behind which can collide with the synthetic
    ``test_sig_*`` keys used here under LRU eviction, making the assertion
    "my most recent entry is in the cache" order-dependent.
    """
    _STRUCTURAL_KERNEL_CACHE.clear()
    yield
    _STRUCTURAL_KERNEL_CACHE.clear()


@pytest.mark.unit
def test_structural_kernel_cache_hit():
    """Two SEDModel instances with identical config share kernels."""
    # Load minimal data
    try:
        ssp = tengri.load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    except FileNotFoundError:
        pytest.skip("SSP data not available")

    filters = tengri.load_filter_set(
        ["hst_f606w", "hst_f775w", "hst_f814w", "hst_f850lp", "vista_ks", "irac_36"]
    )
    obs = tengri.observation.Observation(
        photometry=tengri.observation.Photometry.from_filter_set(filters)
    )

    def make_model():
        spec = tengri.Parameters(
            mean_sfh_type="dpl",
            sfh_dpl_alpha=tengri.Uniform(0.5, 4.0),
            sfh_dpl_beta=tengri.Uniform(0.5, 4.0),
            sfh_dpl_tau_gyr=tengri.Uniform(0.5, 12.0),
            sfh_dpl_log_peak_sfr=tengri.Uniform(-1.0, 2.5),
            met_logzsol=tengri.Uniform(-2.0, 0.2),
            dust_law_bc="calzetti",
            dust_tau_bc=tengri.Uniform(0, 3),
            nebular_ssp=False,
            apply_igm=False,
            redshift=tengri.Fixed(0.5),
        )
        return tengri.SEDModel(spec, ssp, observation=obs)

    m1 = make_model()
    m2 = make_model()

    # Signatures should be identical
    sig1 = m1.compile_signature()
    sig2 = m2.compile_signature()
    assert sig1 == sig2, "Identical configs should have identical signatures"

    # Identity check on the private cache attrs is the only way to prove
    # a cache HIT — there is no public side-channel for "did we reuse the
    # compiled kernels?". This is the test the cache exists for.
    assert m1._compositional_kernels is m2._compositional_kernels, (
        "Compositional kernels not shared across instances"
    )
    assert m1._hybrid_kernels is m2._hybrid_kernels, "Hybrid kernels not shared across instances"


@pytest.mark.unit
def test_structural_kernel_cache_miss_different_config():
    """Different config produces different signatures and separate caches."""
    try:
        ssp = tengri.load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    except FileNotFoundError:
        pytest.skip("SSP data not available")

    filters = tengri.load_filter_set(["hst_f606w", "hst_f775w", "hst_f814w"])
    obs = tengri.observation.Observation(
        photometry=tengri.observation.Photometry.from_filter_set(filters)
    )

    spec1 = tengri.Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=tengri.Uniform(0.5, 4.0),
        sfh_dpl_beta=tengri.Uniform(0.5, 4.0),
        sfh_dpl_tau_gyr=tengri.Uniform(0.5, 12.0),
        sfh_dpl_log_peak_sfr=tengri.Uniform(-1.0, 2.5),
        met_logzsol=tengri.Uniform(-2.0, 0.2),
        dust_law_bc="calzetti",
        dust_tau_bc=tengri.Uniform(0, 3),
        nebular_ssp=False,
        apply_igm=False,
        redshift=tengri.Fixed(0.5),
    )

    spec2 = tengri.Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=tengri.Uniform(0.5, 4.0),
        sfh_dpl_beta=tengri.Uniform(0.5, 4.0),
        sfh_dpl_tau_gyr=tengri.Uniform(0.5, 12.0),
        sfh_dpl_log_peak_sfr=tengri.Uniform(-1.0, 2.5),
        met_logzsol=tengri.Uniform(-2.0, 0.2),
        dust_law_bc="calzetti",
        dust_tau_bc=tengri.Uniform(0, 3),
        nebular_ssp=False,
        apply_igm=True,  # Different from spec1
        redshift=tengri.Fixed(0.5),
    )

    m1 = tengri.SEDModel(spec1, ssp, observation=obs)
    m2 = tengri.SEDModel(spec2, ssp, observation=obs)

    sig1 = m1.compile_signature()
    sig2 = m2.compile_signature()

    assert sig1 != sig2, "Different configs should produce different signatures"
    # Negative identity check — same justification as the positive one above.
    assert m1._compositional_kernels is not m2._compositional_kernels, (
        "Different signatures should have separate kernels"
    )


@pytest.mark.unit
def test_structural_kernel_cache_lru_eviction():
    """LRU eviction works when cache exceeds maxsize."""
    # _STRUCTURAL_KERNEL_CACHE / _MAXSIZE are bound at module import to
    # the real ModelCacheOwner singleton — see the module-level setup.
    _STRUCTURAL_KERNEL_CACHE.clear()

    # Create signatures up to max size
    sigs = []
    for i in range(_STRUCTURAL_KERNEL_MAXSIZE + 2):
        sig = (f"test_sig_{i}",)
        cache = get_structural_kernel_cache(sig)
        sigs.append(sig)
        cache["value"] = i

    # Cache should be bounded at maxsize
    assert len(_STRUCTURAL_KERNEL_CACHE) <= _STRUCTURAL_KERNEL_MAXSIZE, (
        f"Cache size {len(_STRUCTURAL_KERNEL_CACHE)} exceeds maxsize {_STRUCTURAL_KERNEL_MAXSIZE}"
    )

    # Oldest entries should have been evicted (default maxsize=4, so first 2 gone)
    # Check that at least the most recent signature is still there
    assert sigs[-1] in _STRUCTURAL_KERNEL_CACHE, "Most recent entry should be in cache"
    assert _STRUCTURAL_KERNEL_CACHE[sigs[-1]]["value"] == _STRUCTURAL_KERNEL_MAXSIZE + 1

    clear_structural_kernel_cache()


@pytest.mark.unit
def test_clear_structural_kernel_cache():
    """Clear function drops all cached kernels."""
    # Create a test signature and cache
    sig = ("test_sig",)
    cache = get_structural_kernel_cache(sig)
    cache["test_key"] = "test_value"

    # Verify it's there
    assert "test_key" in get_structural_kernel_cache(sig)

    # Clear
    clear_structural_kernel_cache()

    # Verify it's gone
    fresh_cache = get_structural_kernel_cache(sig)
    assert "test_key" not in fresh_cache, "Cache should be empty after clear"


@pytest.mark.unit
def test_gc_clears_structural_kernel_cache():
    """tengri.gc() clears structural kernel cache."""
    # Create a test signature and cache
    sig = ("test_sig",)
    cache = get_structural_kernel_cache(sig)
    cache["test_key"] = "test_value"

    # Verify it's there
    assert "test_key" in get_structural_kernel_cache(sig)

    # Call gc
    tengri.gc()

    # Verify it's gone
    fresh_cache = get_structural_kernel_cache(sig)
    assert "test_key" not in fresh_cache, "gc() should clear structural cache"
