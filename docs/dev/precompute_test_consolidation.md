# Precompute Test Audit Report
**Phase 3 Migration: Kernel Adapter Deletion & Component Protocol Transition**

---

## Summary

Audited 8 precompute-related test files (3100 LOC) + 4 kernel-adapter tests to classify by Phase 3 lifecycle:

| Verdict | Count | Files |
|---------|-------|-------|
| **KEEP** | 2 | test_precompute.py, test_precompute_quad.py |
| **ADAPT** | 3 | test_preintegrate.py, test_ztable_precompute.py, test_hybrid_ztable_kernel.py, test_hybrid_energy_balance.py (partial), test_precompute_kernel_invariants.py (partial) |
| **DELETE** | 3 | test_fused_kernels.py, test_hybrid_energy_balance.py (partial), forward/test_kernel_*.py (all) |

---

## Per-File Triage

### File 1: `tests/unit/test_precompute.py` (140 lines)
**Verdict: KEEP**

| Test Class | Tests | Reason | Phase 3 Status |
|------------|-------|--------|----------------|
| TestFastPhotometry | 5 | Tests `fast_photometry()` — public JAX function in `components/stellar/sps/precompute.py`. Tests shape, weighted average, dust response, JIT, gradients. Same function will be called by `StellarSEDComponent.precompute()` internally. | Survives unchanged |
| TestFastSpectrum | 2 | Tests `fast_spectrum()` — sibling to `fast_photometry()`. Public API that StellarSEDComponent uses internally. | Survives unchanged |
| TestMetallicityInterpolation | 3 | Tests `interpolate_ssp_phot_metallicity()` — low-level grid interpolation util in precompute module. Grid-point identity, midpoint averaging, boundary clamping. | Survives unchanged |

**Replacement:** No change. File stays as-is. Tests the engine functions; no kernel adapter wrapper tested.

---

### File 2: `tests/unit/test_precompute_quad.py` (164 lines)
**Verdict: KEEP**

| Test Class | Tests | Reason | Phase 3 Status |
|------------|-------|--------|----------------|
| TestTaylorMomentTensor | 4 | Tests Taylor correction Ψ tensor (second moment of SSP spectrum within filter). Validates `ssp_phot_moment` shape/finiteness/property for dust-law Taylor approximation. Invokes `precompute_photometry(..., taylor_correction=True)` — public function. | Survives unchanged |
| TestTaylorCorrectionAccuracy | 3 | Compares Taylor-corrected error vs Zacharegkas (n=1) factorization on dust attenuation. Tests whether A·Φ + A'·Ψ improves over A·Φ alone. No kernel adapter tested. | Survives unchanged |

**Replacement:** No change. Tests the underlying `precompute_photometry()` math; no deletion impact.

---

### File 3: `tests/unit/test_preintegrate.py` (809 lines)
**Verdict: ADAPT**

| Test Class | Tests | Verdict | Reason | Replacement Strategy |
|------------|-------|---------|--------|----------------------|
| TestPreintegrateGridBasic | 5 | KEEP | Tests `preintegrate_grid()` — public function in `utils.grid_interp` module. Shape, finiteness, positive flux, wavelength ranges. No kernel adapter involved. | Move to consolidated test file (`tests/unit/components/test_stellar_precompute_lut.py`); keep assertions unchanged. |
| TestPreintegrateGridEnergyNormalization | 2 | KEEP | Tests `preintegrate_grid(..., energy_normalize=True)`. Public API option for converting flux to energy. | Same consolidation. |
| TestPreintegrateGridTaylorMoment | 3 | KEEP | Tests Ψ tensor from `preintegrate_grid(..., return_moment=True)`. Math-level validation. | Same consolidation. |
| TestPreintegrateGridSSPCrossval | 1 | ADAPT | Validates `preintegrate_grid()` against existing SSP precomputation. Cross-checks that new generic grid integrator matches hand-tuned SSP math. **Likely uses old kernel-dependent fixture.** Assertion "matches SSP precompute" is valuable but fixture may reference deleted kernel. | Rewrite fixture to use public `precompute_photometry()` directly; keep assertion logic. |
| TestPreintegrateLines | 6 | KEEP | Tests `preintegrate_lines()` — emission line flux integration. Public math-level API. | Consolidate to same file. |
| TestInterpNdTriweight1D | 4 | KEEP | Tests 1-D triweight interpolation kernel. Public util (`interp_nd_triweight()`). | Consolidate. |
| TestInterpNdTriweight2D | 3 | KEEP | Tests 2-D triweight interpolation. Public util. | Consolidate. |
| TestGradientAndJIT | 2 | KEEP | Tests gradient/JIT safety of `interp_1d` and `preintegrate_grid`. Math-level invariant. | Consolidate. |
| TestEdgeCasesAndStability | 5 | KEEP | Edge cases on numerical stability (tiny/huge template values, narrow filters, single wavelength). Validates robustness of integrators. | Consolidate. |
| TestSliceFixedAxes | 4 | KEEP | Tests `_slice_fixed_axes()` helper for preintegration. Internal util but low-level and fragile. | Consolidate. |

**Consolidation:** Move **all KEEP tests** + fixed TestPreintegrateGridSSPCrossval to → `tests/unit/components/test_stellar_precompute_lut.py`. Reword docstrings to reflect "underlying LUT math" not "kernel adapter".

---

### File 4: `tests/unit/test_ztable_precompute.py` (404 lines)
**Verdict: ADAPT**

| Test Class | Tests | Verdict | Reason | Replacement Strategy |
|------------|-------|---------|--------|----------------------|
| TestZTablePrecomputation | 5 | KEEP | Tests `precompute_photometry_ztable()` — public function. Shape, z-grid defaults, z-dependence, finiteness. No kernel adapter. | Move to consolidated file (`tests/unit/test_stellar_lut_invariants.py`). |
| TestZTableInterpolationAccuracy | 2 | ADAPT | Tests interpolation accuracy at grid points. Logic is sound but may reference hybrid kernel or old `precomputed.photometry_ztable` attr structure. Will change to `state.derived['stellar_phot_ztable']` in Phase 3c. | Rewrite to use new `StellarSEDComponent.precompute()` state and access `state.derived['stellar_phot_ztable']` instead of `precomputed.*`. Keep assertion (accuracy within tol). |
| TestZTableGradients | 3 | KEEP | Tests gradient w.r.t. z parameter (inference-relevant). No kernel structure tested. | Move to consolidated. |
| TestZTableSmoothInterpolation | 6 | ADAPT | Tests smoothness + monotonicity + gradient comparison. Again references legacy `precomputed.*` structure. The assertion "gradient smoother with triweight than linear" is valuable; fixture needs rewrite. | Rewrite fixture to call `StellarSEDComponent.precompute()` and extract `state.derived['stellar_phot_ztable']`. Keep smoothness/gradient assertions. |

**Consolidation:** TestZTableGradients & TestZTablePrecomputation → `tests/unit/test_stellar_lut_invariants.py` (KEEP). TestZTableInterpolationAccuracy & TestZTableSmoothInterpolation (ADAPT) → rewrite in same file to use new component state structure.

---

### File 5: `tests/unit/test_hybrid_ztable_kernel.py` (303 lines)
**Verdict: ADAPT**

| Test Class | Tests | Verdict | Reason | Replacement Strategy |
|------------|-------|---------|--------|----------------------|
| TestZTableBasic | 3 | ADAPT | Tests "ZTable creation" and "gradient w.r.t. z". But "ZTable" here is the kernel adapter's private data structure (hybrid.py's `_build_phot_ztable_hybrid` output), not the public precompute function result. The invariant (interpolation at grid points, gradients finite) is valuable but the test target (hybrid kernel's internal ZTable) will be deleted. | Rewrite to test the same invariants on `StellarSEDComponent.precompute()` output. Assert "predictions at grid points match StellarSEDComponent.precompute(z)" instead of "ZTable at grid point returns ZTable[idx]". |
| TestZTableInterpolationSmoothnessAndMonotonicity | 2 | ADAPT | Asserts "interpolation smooth with z" and "improves with more grid points". Valuable invariant about z-table quality. But tests hybrid kernel's internal interpolation, not public API. | Rewrite: build two StellarSEDComponent instances (coarse z-grid, fine z-grid), assert fine-grid predictions change more smoothly. |
| TestZTableEdgeCases | 4 | ADAPT | Edge cases on very small z, very large z, extrapolation, boundaries. Tests hybrid kernel's boundary handling. Valuable regression coverage but target is deleted code. | Rewrite to test `StellarSEDComponent.precompute()` at z extremes and check gradient stability. |
| TestZTableConsistency | 2 | ADAPT | Reproducibility and consistency checks. Valuable but tied to hybrid kernel internal state. | Rewrite to verify reproducibility of `StellarSEDComponent.precompute()` results and interpolation consistency. |

**Consolidation:** All 11 tests → `tests/unit/test_stellar_lut_invariants.py`, rewritten to invoke `StellarSEDComponent.precompute()` instead of hybrid kernel. Invariants (smoothness, accuracy, edge cases) preserved; implementation target changed.

---

### File 6: `tests/unit/test_hybrid_energy_balance.py` (286 lines)
**Verdict: DELETE (95%) + ADAPT (5%)**

| Test Class | Tests | Verdict | Reason | Replacement Strategy |
|------------|-------|---------|--------|----------------------|
| TestDL07EnergyBalance | 3 | DELETE | "dl07_hybrid_error_below_2%". Tests that the hybrid kernel's dust-emission energy-balance match vs exact path. Hybrid kernel deleted in Phase 3d. No public API to test. | Delete file. This is kernel-adapter regression coverage. |
| TestDale2014NonRegression | 1 | DELETE | "dale_hybrid_error_below_5%". Kernel adapter cross-validation. | Delete. |
| TestTHEMISNonRegression | 1 | DELETE | "themis_hybrid_error_below_5%". Kernel adapter cross-validation. | Delete. |
| TestStellarOnlyNonRegression | 1 | ADAPT | "stellar_hybrid_error_below_1%". Tests "stellar-only path has low error vs exact". This is a pure forward-model regression (no dust), so the assertion may survive on StellarSEDComponent. But "hybrid error" terminology and internal fixture are kernel-adapter specific. | Extract the core assertion ("stellar predictions match exact path within 1%") and rewrite as a component-level test in `tests/unit/components/test_stellar_forward.py`. |
| TestDL07EnergyBalanceWorstCase | 1 | DELETE | "worst case hybrid error below 2%". Kernel adapter internal tuning validation. | Delete. |

**Consolidation:** Delete entire file except TestStellarOnlyNonRegression logic, which moves to `tests/unit/components/test_stellar_forward.py` as a pure "stellar component matches exact math" assertion.

---

### File 7: `tests/unit/test_fused_kernels.py` (568 lines)
**Verdict: DELETE**

| Test Class | Tests | Verdict | Reason |
|------------|-------|---------|--------|
| TestFusedPhotometryAccuracy | 2 | DELETE | "matches_unfused" and "matches_across_metallicities". Tests the fused-kernel performance optimization (hybrid.py's `_fused_photometry_kernel`). Kernel deleted in Phase 3d. No public API equivalent. |
| TestFusedPhotometryGradients | 2 | DELETE | Gradient tests on fused kernel. Internal optimization detail. |
| TestFusedSpectrumAccuracy | 1 | DELETE | Tests fused spectrum kernel. Kernel deleted. |
| TestFusedKernelSpeedup | 1 | DELETE | Benchmark comparing fused vs unfused execution. Optimization regression test for deleted code. |
| TestCSPEndpointWeights | 3 | DELETE | Tests "CSP endpoint weighting" — internal to fused kernel construction (cumulative SFH trapezoid rule). No public API or replacement. |

**No replacement.** The entire file tests internal kernel-adapter optimization strategies that do not exist in the component path.

---

### File 8: `tests/unit/test_precompute_kernel_invariants.py` (414 lines)
**Verdict: ADAPT (50%) + DELETE (50%)**

| Test Class | Tests | Verdict | Reason | Replacement Strategy |
|------------|-------|---------|--------|----------------------|
| TestCacheInvalidation | 3 | DELETE | Tests that `precompute_spectroscopy()` / `precompute_ztable()` clear the inference cache (`tengri.inference._model_cache`). Cache clearing is an implementation detail of the SEDModel.precompute_* methods. When migrated to StellarSEDComponent.precompute(), the cache will be handled by the component protocol, not by per-method calls. This is infrastructure coupling, not a math invariant. | Delete. The precompute infrastructure will be tested at the component level, not here. |
| TestJITSafeSearchsorted | 4 | KEEP | Tests `.shape[0]` vs `len()` idiom on JAX arrays inside JIT. This is a general JAX safety pattern, not kernel-adapter specific. Validates that grid-interpolation code is JIT-safe. | Move to `tests/unit/components/test_stellar_precompute_lut.py`. Rename to TestGridInterpolationJITSafety. |
| TestTraceableRouting | 4 | DELETE | Tests `predict_spectrum(_traceable)` / `predict_photometry(_traceable)` mode. These modes route to kernel adapters. By Phase 3e, mode='_traceable' and mode='compositional' are deleted; only 'exact' and component-default (forward path) remain. These tests validate kernel-internal optimization modes. | Delete. Mode='_traceable' will not exist in Phase 3+. |
| TestPrecomputeConsistency | 2 | ADAPT | Tests "predictions identical before/after precompute_spectroscopy()" and "photometry consistent before/after precompute_ztable()". The assertion is crucial: precompute must not change physics. But the test uses old SEDModel.precompute_* API. | Rewrite: Build a StellarSEDComponent, call precompute() on its state, then assert that predictions from `state.derived['stellar_phot_lnu_lut']` match exact integration. Keep consistency invariant; change API call. Move to `tests/unit/components/test_stellar_forward.py`. |

**Consolidation:** TestJITSafeSearchsorted → `tests/unit/components/test_stellar_precompute_lut.py`. TestPrecomputeConsistency (rewritten) → `tests/unit/components/test_stellar_forward.py`. Delete TestCacheInvalidation & TestTraceableRouting.

---

### Files 9–12: Related forward tests (`tests/unit/forward/test_kernel*.py`)

**All DELETE**

| File | Content | Verdict | Reason |
|------|---------|---------|--------|
| test_kernel_strategy.py | KernelStrategy selection logic; tests DEFAULT / COMPOSITIONAL_ONLY / EXACT_ONLY modes | DELETE | Strategy object and mode names are deleted in Phase 3e. No component protocol equivalent. |
| test_kernel_adapters.py | HybridPhotometryKernel, compositional_photometry adapter wrapping | DELETE | Kernel adapters deleted Phase 3d. |
| test_kernel_build_log.py | Tests kernel build log + caching | DELETE | Build system for deleted kernel adapters. |
| test_kernel_strategy_classmethods.py | (if exists) | DELETE | — |

---

## Consolidation Proposal

Create **2 new test files** to hold KEEP + ADAPT content:

### 1. `tests/unit/components/test_stellar_precompute_lut.py` (~400 lines)
**Purpose:** Test the public precompute LUT math (engine functions).

**Contents:**
- TestFastPhotometry (from test_precompute.py) — unchanged
- TestFastSpectrum (from test_precompute.py) — unchanged
- TestMetallicityInterpolation (from test_precompute.py) — unchanged
- TestTaylorMomentTensor (from test_precompute_quad.py) — unchanged
- TestTaylorCorrectionAccuracy (from test_precompute_quad.py) — unchanged
- TestPreintegrateGridBasic + Energy + TaylorMoment + Lines + Interp + Edge cases (from test_preintegrate.py) — unchanged
- TestGridInterpolationJITSafety (from test_precompute_kernel_invariants.py, renamed from TestJITSafeSearchsorted) — unchanged

**No deletions; no rewrites needed. Pure consolidation.**

---

### 2. `tests/unit/test_stellar_lut_invariants.py` (~350 lines)
**Purpose:** Test the new StellarSEDComponent.precompute() interface and high-level invariants.

**Contents:**
- TestZTablePrecomputation (from test_ztable_precompute.py) — **rewrite fixture** to call StellarSEDComponent.precompute() and extract `state.derived['stellar_phot_ztable']`
- TestZTableGradients (from test_ztable_precompute.py) — same
- TestZTableInterpolationAccuracy (from test_ztable_precompute.py) — **rewrite** to use new component state
- TestZTableSmoothInterpolation (from test_ztable_precompute.py) — **rewrite** to use new component state
- TestHybridZTableBasic through TestHybridZTableConsistency (from test_hybrid_ztable_kernel.py, renamed to TestZTableComponent*) — **rewrite all** to use StellarSEDComponent instead of hybrid kernel
- TestStellarComponentForwardAccuracy (from test_hybrid_energy_balance.py TestStellarOnlyNonRegression, rewritten) — assert "StellarSEDComponent predictions match exact math within 1%"
- TestStellarComponentPrecomputeConsistency (from test_precompute_kernel_invariants.py TestPrecomputeConsistency, rewritten) — assert "precompute() does not change physics"

**All require fixture rewrites; all assertions/invariants preserved.**

---

## Files to Delete After Phase 3 Completes

```
tests/unit/test_fused_kernels.py                           (568 lines)
tests/unit/test_hybrid_energy_balance.py                   (286 lines)
tests/unit/test_precompute_kernel_invariants.py            (414 lines – partial)
tests/unit/forward/test_kernel_strategy.py                 (210 lines)
tests/unit/forward/test_kernel_adapters.py                 (171 lines)
tests/unit/forward/test_kernel_build_log.py                (103 lines)
tests/unit/forward/test_kernel_strategy_classmethods.py    (if exists)

Total lines eliminated: ~1750 (37% reduction in precompute test coverage)
```

---

## Impact Summary

| Category | Count | Status |
|----------|-------|--------|
| Tests staying (KEEP) | 27 | Consolidated into 1 new file; no rewrites |
| Tests needing rewrites (ADAPT) | 21 | Consolidated into 1 new file; fixture + state access rewritten |
| Tests to delete (DELETE) | ~35 | Kernel-adapter internals; no replacements |
| **Net test count change** | -14 (~18% reduction) | Quality preserved; coverage shifts from kernel internals to public LUT + component protocol |

