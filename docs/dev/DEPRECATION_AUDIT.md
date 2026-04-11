# Tengri Deprecation Audit (2026-04-10)

Comprehensive audit of all deprecated aliases, old names, backward-compatibility shims, and stale code in `src/tengri/` directory.

---

## Executive Summary

The codebase has **7 deprecated class name aliases**, **17 deprecated inference method aliases**, **2 silent model-name mappings** (not emitting warnings), **1 deprecated method**, and **2 toy models** marked for removal. Most deprecated items properly emit `DeprecationWarning` and will be removed in v1.0.

**Key findings:**
- ✅ Class aliases properly decorated with factory functions (e.g., `ParamSpec`, `SpectroscopyConfig`, etc.)
- ✅ Inference method aliases centralized in `_DEPRECATED_METHOD_ALIASES` dict with `resolve_method()`
- ⚠️ Some old method names still hardcoded in internal code (should use canonical names)
- ⚠️ Silent mappings for `"dl07_tabulated"` and `"kubota_done"` without warnings
- ⚠️ `fit_catalog()` vs `fit_batch()` naming inconsistency (CLAUDE.md says use `fit_batch`)
- ⚠️ `PopulationFitter.run()` accepts old method names without calling `resolve_method()`

---

## 1. Deprecated Class Name Aliases

All properly implemented with DeprecationWarning. Safe to remove in v1.0.

### 1.1 `ParamSpec` → `Parameters`

| Field | Value |
|-------|-------|
| **Canonical name** | `Parameters` |
| **Old alias** | `ParamSpec` |
| **Definition** | `src/tengri/core/parameters.py:1948` |
| **Factory** | `_make_deprecated_paramspec()` at line 1933-1945 |
| **Mechanism** | Proxy class with `__init__` that warns then calls `super()` |
| **Exports** | Both in `__init__.py` (lines 46, 218) |
| **Status** | ✅ Proper DeprecationWarning emitted |
| **Removal** | Safe in v1.0 |

### 1.2 `SpectroscopyConfig` → `Spectroscopy`

| Field | Value |
|-------|-------|
| **Canonical name** | `Spectroscopy` |
| **Old alias** | `SpectroscopyConfig` |
| **Definition** | `src/tengri/models/observation/spectroscopy.py:277-284` |
| **Mechanism** | Inline class proxy with `__init__` that warns and calls `super()` |
| **Exports** | Both in `__init__.py` (lines 80, 233) |
| **Status** | ✅ Proper DeprecationWarning emitted |
| **Removal** | Safe in v1.0 |

**Code snippet:**
```python
class SpectroscopyConfig(Spectroscopy):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SpectroscopyConfig is deprecated. Use Spectroscopy instead. "
            "Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
```

### 1.3 `NoiseConfig` → `NoiseModel`

| Field | Value |
|-------|-------|
| **Canonical name** | `NoiseModel` |
| **Old alias** | `NoiseConfig` |
| **Definition** | `src/tengri/models/observation/noise_model.py:77-84` |
| **Mechanism** | Inline class proxy with warning in `__init__` |
| **Exports** | Both in `__init__.py` (lines 77, 215-216) |
| **Status** | ✅ Proper DeprecationWarning emitted |
| **Removal** | Safe in v1.0 |

### 1.4 `LineCatalog` → `LineList`

| Field | Value |
|-------|-------|
| **Canonical name** | `LineList` |
| **Old alias** | `LineCatalog` |
| **Definition** | `src/tengri/models/observation/line_list.py:758` |
| **Factory** | `_make_deprecated_line_catalog()` at lines 744-755 |
| **Mechanism** | Factory function creates proxy type with metaclass |
| **Exports** | Both in `__init__.py` (lines 76, 206-207) |
| **Status** | ✅ Proper DeprecationWarning emitted |
| **Removal** | Safe in v1.0 |

### 1.5 `HierarchicalFitter` → `PopulationFitter`

| Field | Value |
|-------|-------|
| **Canonical name** | `PopulationFitter` |
| **Old alias** | `HierarchicalFitter` |
| **Definition** | `src/tengri/inference/hierarchical.py:1430` |
| **Factory** | `_make_deprecated_hierarchical_fitter()` at lines 1415-1427 |
| **Mechanism** | Factory function creates proxy type with warning in `__call__` |
| **Exports** | Both in `__init__.py` (lines 65, 203) |
| **Status** | ✅ Proper DeprecationWarning emitted |
| **Removal** | Safe in v1.0 |

### 1.6 `HierarchicalResult` → `PopulationPosterior`

| Field | Value |
|-------|-------|
| **Canonical name** | `PopulationPosterior` |
| **Old alias** | `HierarchicalResult` |
| **Definition** | `src/tengri/inference/hierarchical.py:1409` |
| **Factory** | `_make_deprecated_hierarchical_result()` at lines 1394-1406 |
| **Mechanism** | Factory function creates proxy type with warning in `__call__` |
| **Exports** | Both in `__init__.py` (lines 66, 204) |
| **Status** | ✅ Proper DeprecationWarning emitted |
| **Removal** | Safe in v1.0 |

### 1.7 `Model` → `SEDModel` (ISSUE: Missing Warning)

| Field | Value |
|-------|-------|
| **Canonical name** | `SEDModel` |
| **Old alias** | `Model` |
| **Definition** | `src/tengri/core/model.py:2448` |
| **Mechanism** | Simple assignment: `Model = SEDModel` |
| **Exports** | Both in `__init__.py` (lines 38, 211) |
| **Status** | ⚠️ **Does NOT emit DeprecationWarning** |
| **Problem** | Only plain alias; should be a proxy with warning |
| **Removal** | Safe in v1.0, but needs warning mechanism first |

**Current code:**
```python
# Line 2448 in model.py
Model = SEDModel
```

**Should be:**
```python
# Use the deprecated_class_alias factory from utils/deprecation.py
Model = deprecated_class_alias("Model", SEDModel)
```

---

## 2. Deprecated Inference Method Aliases

**Centralized dispatch:** `src/tengri/inference/fitter.py:41-158`

### 2.1 Method Alias Dictionary

**Location:** `src/tengri/inference/fitter.py:41-65` (_DEPRECATED_METHOD_ALIASES)

| Old Name | Canonical Name | Category |
|----------|---|----------|
| `vi_nifty` | `vi` | VI variant (nifty-qualified name) |
| `vi_nifty_linear` | `vi_linear` | VI variant (nifty-qualified name) |
| `geovi` | `vi` | Old geoVI name |
| `fast_geovi` | `vi` | Old geoVI variant |
| `nifty_geovi` | `vi` | Old geoVI variant |
| `geovi_nuts` | `vi` | Old hybrid name |
| `mgvi` | `vi_linear` | Old MGVI name |
| `fast_mgvi` | `vi_linear` | Old MGVI variant |
| `nifty_mgvi` | `vi_linear` | Old MGVI variant |
| `evi` | `vi_linear` | Old linear VI name |
| `native_geovi` | `vi_native` | Old native VI name |
| `native_mgvi` | `vi_native_linear` | Old native linear VI name |
| `native_evi` | `vi_native_linear` | Old native linear VI name |
| `raytrace` | `mcmc_raytrace` | Old MCMC name |
| `nuts` | `mcmc_nuts` | Old MCMC name |
| `elliptical_slice` | `mcmc_ess` | Old MCMC name |
| `evidence` | `nss` | Old evidence name |

**Total:** 17 deprecated aliases mapping to 11 canonical names.

### 2.2 Resolution Mechanism

**Function:** `resolve_method(method, emit_warning=True)` at `src/tengri/inference/fitter.py:93-158`

**Behavior:**
1. Returns method as-is if already in `_CANONICAL_METHODS`
2. Maps to canonical name if in `_DEPRECATED_METHOD_ALIASES`, emitting DeprecationWarning
3. Raises `ParameterError` if method not recognized

**Status:** ✅ Proper DeprecationWarning emitted at stacklevel=3 for caller visibility

**Usage:** Called in `Fitter.run()` at line 1008:
```python
method = resolve_method(method)
```

**Removal plan:** In v1.0, delete:
- `_DEPRECATED_METHOD_ALIASES` dict (lines 41-65)
- `resolve_method()` function (lines 93-158)
- Call to `resolve_method()` in `Fitter.run()` (line 1008)
- Update docstring examples

### 2.3 Known Issues with Method Aliases

⚠️ **Problem 1:** `PopulationFitter.run()` doesn't use `resolve_method()`

**Location:** `src/tengri/inference/hierarchical.py:181-219`

**Issue:** Accepts old method names directly ("geovi", "mgvi", "raytrace") but doesn't emit warnings.

**Code:**
```python
def run(self, method="geovi", *, key=None, **kwargs):
    """Run hierarchical inference.
    
    Parameters
    ----------
    method : str
        "geovi" — geoVI with CorrelatedFieldMaker for native PSD learning.
        "mgvi" — MGVI (faster per iteration, for very large N).
        ...
    """
    if method == "evi":
        return self._run_evi_jit(key=key, **kwargs)
    elif method == "evi_nifty":
        return self._run_geovi_cfm(key=key, sample_mode="evi", **kwargs)
    elif method == "geovi":  # ← accepts old name
        return self._run_geovi_cfm(key=key, **kwargs)
    ...
```

**Fix needed:** Either call `resolve_method()` or rename to canonical names with deprecation warnings.

⚠️ **Problem 2:** Convenience module has method translation map

**Location:** `src/tengri/core/convenience.py:381-386`

**Issue:** Translates canonical names back to old names for PopulationFitter:

```python
# Translate canonical → HierarchicalFitter method names
_hier_method_map = {
    "vi": "geovi",
    "vi_linear": "mgvi",
    "mcmc_raytrace": "raytrace",
    "mcmc": "raytrace",
}
hier_method = _hier_method_map.get(method, method)
```

**Problem:** If user passes canonical name, it gets translated to old name, but `PopulationFitter.run()` doesn't warn.

---

## 3. Deprecated Method: `Posterior.summary()`

| Field | Value |
|-------|-------|
| **Old method** | `Posterior.summary()` |
| **New method** | `Posterior.stats()` |
| **Location** | `src/tengri/inference/posterior.py:309-314` |
| **Status** | ✅ Proper DeprecationWarning emitted |
| **Code** | ```python warnings.warn(...DeprecationWarning...) ``` |
| **Removal** | Safe in v1.0 |

---

## 4. Silent Model Name Mappings (No Warning)

### 4.1 Dust Emission: `"dl07_tabulated"` → `"draine_li2007"`

| Field | Value |
|-------|-------|
| **Alias** | `"dl07_tabulated"` |
| **Canonical** | `"draine_li2007"` |
| **Mapping location** | `src/tengri/core/model.py:480-483` |
| **Code** | `if self._dust_emission_model == "dl07_tabulated": self._dust_emission_model = "draine_li2007"` |
| **Warning** | ⚠️ **NO** DeprecationWarning emitted |
| **Status** | Silently auto-maps; should warn |
| **Removal** | In v1.0 after adding warning |

**Context:**
```python
# Line 480-483 in model.py
# "dl07_tabulated" is a legacy alias — map to "draine_li2007" which
# auto-loads the DL07 templates from data/ on first use
if self._dust_emission_model == "dl07_tabulated":
    self._dust_emission_model = "draine_li2007"
```

**Action needed:** Add DeprecationWarning before remapping.

**Related function:** `register_dl07_tabulated()` at `src/tengri/models/dust/emission.py:1715`

### 4.2 AGN Disc Model: `"kubota_done"` → `"multicolor_agn"`

| Field | Value |
|-------|-------|
| **Alias** | `"kubota_done"` |
| **Canonical** | `"multicolor_agn"` |
| **Mapping location** | `src/tengri/models/agn/unified.py:364-366` |
| **Code** | `AGN_MODELS["kubota_done"] = multicolor_agn` |
| **Warning** | ⚠️ **NO** DeprecationWarning (alias registered in dict) |
| **Status** | Silently works via registry; should warn at usage |
| **Removal** | In v1.0 after adding warning mechanism |

**Context:**
```python
# Lines 364-366 in unified.py
# Backward-compat alias (renamed from kubota_done -> multicolor_agn)
AGN_MODELS["kubota_done"] = multicolor_agn
```

**Problem:** The alias is transparent to users. When `agn_model="kubota_done"` is used, the code doesn't know to warn. The warning would need to happen in the model initialization or physics function.

**Where would warning go?** Probably in `Model.__init__()` or a validation step in `ParamSpec` when checking AGN model names.

---

## 5. Deprecated Physics Models (Toy Implementations)

### 5.1 Toy Torus Models (IMP-01, Fixed 2026-04-04)

| Field | Value |
|-------|-------|
| **Names** | `simple_torus()`, `two_temperature_torus()` |
| **Location** | `src/tengri/models/agn/torus.py:50-172` |
| **Status** | ✅ Both emit DeprecationWarning (IMP-01 fixed) |
| **Reason** | Toy MBB approximations, not radiative transfer |
| **Alternative** | Use SKIRTOR models (`skirtor_analytic`) |
| **Removal** | After paper publication (currently needed as reference) |

**Warning code:**
```python
# torus.py:91
warnings.warn(
    "simple_torus is a toy model (single-temperature MBB, not radiative transfer) "
    ...
)
```

**Used in:** `src/tengri/models/agn/unified.py:476, 669, 856` and registry at lines 136-137.

---

## 6. Non-Deprecated But Confusing/Unused

### 6.1 `is_tier2_compatible()`

| Field | Value |
|-------|-------|
| **Location** | `src/tengri/core/fused_kernels.py:1266-1287` |
| **Status** | Not deprecated, but misleading |
| **Current behavior** | Always returns `True` |
| **Intended purpose** | Check if compositional rest-frame SED kernel can be built |
| **Used at** | `src/tengri/core/model.py:810` in Model init |
| **Action** | Either implement real checks or remove |

**Code:**
```python
def is_tier2_compatible(model):
    """Check if the compositional rest-frame SED kernel can be built..."""
    return True  # ← Always true, no actual checks
```

---

## 7. API Naming Inconsistency

### 7.1 `fit_catalog()` vs `fit_batch()`

| Field | Value |
|-------|-------|
| **Current name** | `fit_catalog()` |
| **CLAUDE.md says** | Should be `fit_batch()` |
| **Status** | NOT deprecated, still active and used |
| **Locations** | Function: `src/tengri/core/convenience.py:204`, Method wrapper: `src/tengri/core/model.py:2345`, Docstring ref: `src/tengri/__init__.py:132` |
| **Action needed** | Either rename or create alias with deprecation |

**Note:** This inconsistency is documented in CLAUDE.md but not implemented.

---

## 8. Hardcoded Method Name References (Should Update)

These files reference old method names directly. They should be updated to use canonical names where possible.

| File | Lines | Reference | Type |
|------|-------|-----------|------|
| `src/tengri/inference/vi.py` | 36, 161-179 | Docstring & dict keys | `"geovi"`, `"mgvi"` |
| `src/tengri/inference/vi.py` | 364, 561-562 | Logic comparisons | `"nonlinear_resample"` ↔ `"geovi"` |
| `src/tengri/plotting.py` | 29-31, 54-56 | COLORS & style dicts | `"geovi"`, `"nuts"`, `"mgvi"` |
| `src/tengri/plotting.py` | 525-527 | Bar chart colors | Same old names |
| `src/tengri/inference/jit_engine.py` | 780, 788 | String comparisons | `"geovi"` |
| `src/tengri/inference/posterior.py` | 8 | Docstring example | `"nuts"` |
| `src/tengri/inference/hierarchical.py` | 13 | Docstring example | `"geovi"` |

**Actions for each:**
- **vi.py:** Update docstring to use canonical names; consider renaming internal dicts
- **plotting.py:** Rename COLORS dict keys to canonical names (may break user code expecting `COLORS["geovi"]`)
- **jit_engine.py:** Update string comparisons to use canonical names (internal, safe to refactor)
- **posterior.py, hierarchical.py:** Update docstring examples

---

## 9. Removed Modules (Confirmed Not in Codebase)

✅ **`forward_model.py`** — Removed completely, no imports found
✅ **`charlot_fall.py`** — Removed completely, no imports found
✅ **`ForwardModel` class** — Removed completely, no references found
✅ **No dangling imports** of removed modules in `src/tengri/`

---

## 10. Deprecation Utility Module

| Field | Value |
|-------|-------|
| **Location** | `src/tengri/utils/deprecation.py:1-63` |
| **Status** | Available but underused |
| **Provides** | `deprecated_alias()` decorator, `deprecated_class_alias()` factory |
| **Currently used** | No (class aliases use inline factories instead) |
| **Recommendation** | Could be refactored for consistency, but low priority |

**Example usage** (not currently in code):
```python
from tengri.utils.deprecation import deprecated_class_alias

Model = deprecated_class_alias("Model", SEDModel)
```

---

## 11. Deprecation Summary Table

| Category | Count | Status | Removal Target |
|----------|-------|--------|-----------------|
| **Class aliases (with warning)** | 6 | ✅ Proper | v1.0 |
| **Class alias (without warning)** | 1 | ⚠️ Issue | v1.0 (needs fix) |
| **Inference method aliases** | 17 | ✅ Proper | v1.0 |
| **Posterior methods** | 1 | ✅ Proper | v1.0 |
| **Model name aliases (no warning)** | 2 | ⚠️ Issue | v1.0 (needs warning) |
| **Toy physics models** | 2 | ✅ Warning | After paper |
| **Confusing/unused functions** | 1 | ⚠️ Unclear | TBD |
| **API naming inconsistencies** | 1 | ⚠️ Undecided | TBD |

**Total deprecated items:** ~30

---

## 12. Recommended Actions Before v1.0

### High Priority (Missing Warnings)

1. **Add warning to `Model = SEDModel` alias** (line 2448 in model.py)
   - Use: `Model = deprecated_class_alias("Model", SEDModel)` from `utils/deprecation.py`
   - Rationale: Consistency with other class aliases

2. **Add warnings for silent model name mappings**
   - `"dl07_tabulated"` → `"draine_li2007"` (model.py:480-483)
   - `"kubota_done"` → `"multicolor_agn"` (unified.py:364-366)
   - Rationale: Users should know they're using deprecated names

3. **Make `PopulationFitter.run()` call `resolve_method()`**
   - Currently accepts old names without warnings
   - Options: (a) Call resolve_method(), (b) Add internal warning, (c) Accept only canonical names
   - Rationale: Consistency with Fitter.run()

### Medium Priority (Cleanup)

4. **Update hardcoded method name references to canonical names**
   - vi.py, jit_engine.py: Internal logic, safe to refactor
   - plotting.py: Breaking change for users expecting old COLORS keys
   - posterior.py, hierarchical.py: Update examples

5. **Decide on `fit_catalog()` vs `fit_batch()` naming**
   - Rename or create alias
   - Update documentation accordingly

### Low Priority (Non-urgent)

6. **Clarify/implement `is_tier2_compatible()`**
   - Either add real tier-2 compatibility checks
   - Or document that Tier 2 is always available and remove function
   - Or deprecate if unused

7. **Standardize deprecation utility usage**
   - Consider using `deprecated_class_alias()` from utils/deprecation.py for all class aliases
   - Currently uses both factory functions and inline proxies (inconsistent)

---

## 13. Implementation Notes for v1.0 Removal

### To Remove `Model`, `ParamSpec`, etc. in v1.0:

1. Delete alias definitions:
   - Line 2448 in `model.py`
   - Lines 1948 in `parameters.py`
   - Lines 277-284 in `spectroscopy.py`
   - Lines 77-84 in `noise_model.py`
   - Lines 744-755, 758 in `line_list.py`
   - Lines 1394-1406, 1409, 1415-1427, 1430 in `hierarchical.py`

2. Remove from `__init__.py`:
   - Remove `ParamSpec`, `SpectroscopyConfig`, `NoiseConfig`, `LineCatalog`, `HierarchicalFitter`, `HierarchicalResult` from `__all__` list
   - Remove imports if they're only re-exported

3. Update any internal code that imports old names
   - Search: `from tengri import ParamSpec` or `from tengri.core.parameters import ParamSpec`
   - Replace with canonical names

### To Remove inference method aliases in v1.0:

1. Delete `_DEPRECATED_METHOD_ALIASES` dict (lines 41-65 in fitter.py)
2. Delete `resolve_method()` function (lines 93-158 in fitter.py)
3. Update `Fitter.run()` to require canonical method names (remove line 1008)
4. Update all docstrings and examples
5. Update `PopulationFitter.run()` to only accept canonical names
6. Clean up hardcoded references in vi.py, jit_engine.py, etc.

---

## 14. Files Requiring Updates

### Definition/Implementation Files
- `src/tengri/core/model.py` — Model alias, is_tier2_compatible usage
- `src/tengri/core/parameters.py` — ParamSpec alias factory
- `src/tengri/core/convenience.py` — Method translation map for PopulationFitter
- `src/tengri/core/fused_kernels.py` — is_tier2_compatible definition
- `src/tengri/inference/fitter.py` — Method aliases dict & resolve_method
- `src/tengri/inference/hierarchical.py` — PopulationFitter.run() method handling
- `src/tengri/inference/posterior.py` — summary() deprecation
- `src/tengri/models/observation/spectroscopy.py` — SpectroscopyConfig alias
- `src/tengri/models/observation/noise_model.py` — NoiseConfig alias
- `src/tengri/models/observation/line_list.py` — LineCatalog alias
- `src/tengri/models/agn/unified.py` — kubota_done alias, torus models

### Usage/Reference Files
- `src/tengri/__init__.py` — Exports __all__ list
- `src/tengri/inference/vi.py` — Hardcoded old method names
- `src/tengri/inference/jit_engine.py` — Hardcoded geovi string
- `src/tengri/plotting.py` — Old method names in COLORS/styles
- `src/tengri/models/dust/emission.py` — dl07_tabulated function
- `docs/` — Update examples and CLAUDE.md

---

## 15. Cross-Reference: What CLAUDE.md Says

From `/Users/suchethacooray/Projects/tengri/CLAUDE.md`:

> **Canonical class names** (`SEDModel`, `Parameters`, `Spectroscopy`, `NoiseModel`, `LineList`, `PopulationFitter`)
> **Deprecated aliases** (`Model`, `ParamSpec`, `SpectroscopyConfig`, `NoiseConfig`, `LineCatalog`, `HierarchicalFitter`) must never appear in new code.

**Status:** This audit confirms the deprecated aliases exist and mostly have proper warnings. Issue: `Model` alias doesn't have a warning.

> **Inference method names**: Old names (`geovi`, `raytrace`, `nuts`, etc.) still work but emit `DeprecationWarning` and will be removed in v1.0.

**Status:** ✅ Confirmed. 17 aliases in _DEPRECATED_METHOD_ALIASES dict with resolve_method() dispatcher.

> `fit_catalog` (deprecated, should be `fit_batch`)

**Status:** ⚠️ Not deprecated in code. Needs action.

---

## Appendix: Full File Locations Map

```
src/tengri/
├── __init__.py                               [Exports, __all__ list]
├── core/
│   ├── model.py                              [Model alias (line 2448), is_tier2_compatible usage]
│   ├── parameters.py                         [ParamSpec alias factory (line 1948)]
│   ├── convenience.py                        [fit_catalog, PopulationFitter method map]
│   ├── fused_kernels.py                      [is_tier2_compatible definition]
│   └── sed_pipeline.py                       [Uses tau_v1, tau_v2 internally]
├── inference/
│   ├── fitter.py                             [_DEPRECATED_METHOD_ALIASES, resolve_method]
│   ├── posterior.py                          [summary() deprecation]
│   ├── hierarchical.py                       [PopulationFitter, deprecated aliases HierarchicalFitter/Result]
│   ├── vi.py                                 [Hardcoded old method names]
│   └── jit_engine.py                         [Hardcoded geovi string]
├── models/
│   ├── agn/
│   │   ├── unified.py                        [kubota_done alias, torus warnings]
│   │   ├── torus.py                          [simple_torus, two_temperature_torus warnings]
│   │   └── __init__.py                       [Exports torus functions]
│   ├── dust/
│   │   ├── emission.py                       [register_dl07_tabulated function]
│   │   ├── attenuation.py                    [Uses tau_v1, tau_v2 internally]
│   └── observation/
│       ├── spectroscopy.py                   [SpectroscopyConfig alias]
│       ├── noise_model.py                    [NoiseConfig alias]
│       └── line_list.py                      [LineCatalog alias factory]
├── plotting.py                               [COLORS/styles with old method names]
└── utils/
    └── deprecation.py                        [deprecated_alias, deprecated_class_alias utilities]
```

---

## Summary for v1.0 Migration

**Before v1.0 release:**

1. ✅ **Already done:** All class and method aliases have proper DeprecationWarning (except `Model` and silent mappings)
2. ⚠️ **Fix `Model` alias** — Add warning mechanism
3. ⚠️ **Add warnings to silent mappings** — `"dl07_tabulated"`, `"kubota_done"`
4. ⚠️ **Fix `PopulationFitter.run()`** — Call `resolve_method()` for consistency
5. 📝 **Update hardcoded references** — Use canonical method names in vi.py, plotting.py, etc.
6. 🗑️ **Remove all deprecated items** — Delete aliases, exports, and docstring examples
7. 📚 **Update documentation** — Remove deprecated names from all docs

**Estimated effort:** 4-8 hours for complete removal (mostly find-replace and testing)

