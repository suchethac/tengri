# Naming Contract Implementation Plan

> **Date:** 2026-04-05
> **Authority:** `docs/dev/NAMING_CONTRACT.md`
> **Purpose:** Step-by-step plan for subagents to execute the naming overhaul.
> **Constraint:** Naming-only refactor. No logic changes. All 1764+ tests must pass after each phase.
> **Cross-reference:** Class names, verb conventions, and file paths are defined in the Naming Contract. This document is subordinate; if it conflicts with the Contract, the Contract wins.

---

## Phase 0: Pre-flight

Before any rename, establish the infrastructure that later phases depend on.

### 0.1 Create the exception hierarchy (Contract §8)

**File:** `src/tengri/core/exceptions.py` (new file)

```python
class TengriError(Exception):
    """Base exception for all Tengri errors."""

class ParameterError(TengriError):
    """Invalid parameter names, values, or conflicts."""

class ConfigError(TengriError):
    """Invalid Config construction or missing fields."""

class BackendError(TengriError):
    """Backend initialization or computation failure."""

class InferenceError(TengriError):
    """Sampler/optimizer failures (convergence, NaN, etc.)."""

class TengriIOError(TengriError, OSError):
    """File I/O, missing data files, format mismatch."""
```

Export from `src/tengri/core/__init__.py` and `src/tengri/__init__.py`.

**Note:** Named `TengriIOError` (not `IOError`) to avoid shadowing the builtin. Inherits from both `TengriError` and `OSError` so existing `except OSError` handlers still catch it.

### 0.2 Add a deprecation helper utility

**File:** `src/tengri/utils/deprecation.py` (new file)

```python
import warnings
import functools

def deprecated_alias(canonical_name: str, remove_in: str = "1.0"):
    """Decorator that emits DeprecationWarning when the old name is called."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated, use {canonical_name} instead. "
                f"Will be removed in tengri v{remove_in}.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def deprecated_class_alias(old_name: str, new_cls, remove_in: str = "1.0"):
    """Return a class alias that warns on instantiation."""
    class AliasType(type):
        def __call__(cls, *args, **kwargs):
            warnings.warn(
                f"{old_name} is deprecated, use {new_cls.__name__} instead. "
                f"Will be removed in tengri v{remove_in}.",
                DeprecationWarning,
                stacklevel=2,
            )
            return new_cls(*args, **kwargs)

        def __instancecheck__(cls, instance):
            return isinstance(instance, new_cls)

        def __subclasscheck__(cls, subclass):
            return issubclass(subclass, new_cls)

    return AliasType(old_name, (new_cls,), {})
```

This avoids duplicating warning boilerplate across every module. All phases below use these helpers.

### 0.3 Verify baseline

```bash
pytest tests/ -q
ruff check src/ tests/
ruff format --check src/ tests/
```

Record the baseline test count. Every subsequent phase must match it.

---

## Phase 1: `Model` → `SEDModel` (Contract §2)

### 1.1 Rename the class definition

**File:** `src/tengri/core/model.py`

- Rename `class Model` to `class SEDModel`.
- Update the class docstring: replace self-references from `Model` to `SEDModel`.
- Add deprecated alias at module level:

```python
class SEDModel:
    """The forward SED model. Formerly ``Model``."""
    ...

# Deprecated alias — emits warning on instantiation.
# Simple assignment kept alongside for isinstance/import compatibility.
Model = SEDModel
```

The simple assignment (`Model = SEDModel`) is sufficient for internal backward compatibility. The deprecation warning is emitted by `__init__.py`-level `__getattr__` (see 1.2).

### 1.2 Update `__init__.py` exports

**File:** `src/tengri/__init__.py`

```python
from tengri.forward.sed_model import SEDModel, PriorPredictive

# Deprecated alias with warning
def __getattr__(name):
    if name == "Model":
        import warnings
        warnings.warn(
            "tengri.Model is deprecated, use tengri.SEDModel instead. "
            "Will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SEDModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

Update `__all__`: add `"SEDModel"`, keep `"Model"` for discoverability.

### 1.3 Update all internal imports

Search targets:
```bash
grep -rn "from tengri.forward.sed_model import.*\bModel\b" src/ tests/
grep -rn "from tengri import.*\bModel\b" src/ tests/
grep -rn "\bModel\b" src/tengri/ --include="*.py" | grep -v "SEDModel\|NoiseModel\|__pycache__"
```

Files likely affected (verify with grep — do not assume this list is exhaustive):
- `src/tengri/core/mock.py`
- `src/tengri/core/prediction.py`
- `src/tengri/core/display.py`
- `src/tengri/inference/fitter.py`
- `src/tengri/inference/hierarchical.py`
- `src/tengri/inference/jit_engine.py`
- `src/tengri/inference/sbi.py`
- All test files referencing `Model`

### 1.4 Update type annotations and docstrings

Any type hint `model: Model` → `model: SEDModel`. Any docstring mentioning "a `Model` instance" → "an `SEDModel` instance".

### 1.5 Verify

```bash
pytest tests/ -q
ruff check src/ tests/
ruff format --check src/ tests/
python -c "from tengri import SEDModel; print(SEDModel)"
python -c "from tengri import Model"  # should emit DeprecationWarning but succeed
python -c "from tengri.forward.sed_model import Model; from tengri.forward.sed_model import SEDModel; assert Model is SEDModel"
```

---

## Phase 2: Flip canonical direction for existing aliases (Contract §2)

Each sub-phase is independent. Run verification after each.

### 2.1 `ParamSpec` → `Parameters`

**File:** `src/tengri/core/param_spec.py`

- Rename `class ParamSpec` → `class Parameters`.
- Add `ParamSpec = Parameters` at module level.
- **Rename file:** `param_spec.py` → `parameters.py` (Contract §2 file path rule).
- Update `core/__init__.py` import path.
- Update all internal imports to `from tengri.parameters.parameters import Parameters`.
- Keep `ParamSpec` in `__all__` for backward compatibility.

**Caution:** The file rename means every `from tengri.core.param_spec import ...` in the codebase must be updated in the same commit. Use grep:
```bash
grep -rn "param_spec" src/ tests/
```

### 2.2 `SpectroscopyConfig` → `Spectroscopy`

**File:** `src/tengri/models/observation/spectroscopy_config.py`

- Rename `class SpectroscopyConfig` → `class Spectroscopy`.
- Add `SpectroscopyConfig = Spectroscopy` at module level.
- **Rename file:** `spectroscopy_config.py` → `spectroscopy.py`.
- Update all internal imports.

### 2.3 `NoiseConfig` → `NoiseModel`

**File:** `src/tengri/models/observation/noise_config.py`

- Rename `class NoiseConfig` → `class NoiseModel`.
- Add `NoiseConfig = NoiseModel` at module level.
- **Rename file:** `noise_config.py` → `noise_model.py`.
- Update all internal imports.

### 2.4 `LineCatalog` → `LineList`

**File:** `src/tengri/models/observation/line_catalog.py`

- Rename `class LineCatalog` → `class LineList`.
- Add `LineCatalog = LineList` at module level.
- **Rename file:** `line_catalog.py` → `line_list.py`.
- Update all internal imports.

### 2.5 `HierarchicalFitter` → `PopulationFitter`, `HierarchicalResult` → `PopulationPosterior`

**File:** `src/tengri/inference/hierarchical.py`

- Rename both classes.
- Add both aliases at module level.
- File stays as `hierarchical.py` (Contract §2 lists `inference/hierarchical.py` as the defining module for both).
- Update all internal imports.

### 2.6 Verify after each sub-phase

```bash
pytest tests/ -q
ruff check src/ tests/
ruff format --check src/ tests/
```

---

## Phase 3: Function verb standardization (Contract §4)

### 3.1 AGN compute functions

| File | Current | Canonical | Alias added |
|------|---------|-----------|-------------|
| `models/agn/blr.py` | `blr_emission()` | `compute_blr_sed()` | `blr_emission = compute_blr_sed` |
| `models/agn/nlr.py` | `nlr_emission()` | `compute_nlr_sed()` | `nlr_emission = compute_nlr_sed` |
| `models/agn/qsogen.py` | `qsogen_sed()` | `compute_qsogen_sed()` | `qsogen_sed = compute_qsogen_sed` |

### 3.2 Nebular compute functions

| File | Current | Canonical | Alias added |
|------|---------|-----------|-------------|
| `models/nebular/shock.py` | `shock_emission_sed()` | `compute_shock_sed()` | `shock_emission_sed = compute_shock_sed` |

### 3.3 Registry lookups (`get_*` → `resolve_*`)

| File | Current | Canonical | Alias added |
|------|---------|-----------|-------------|
| `models/dust/attenuation.py` | `get_dust_law()` | `resolve_dust_law()` | `get_dust_law = resolve_dust_law` |
| `models/agn/unified.py` | `get_agn_model()` | `resolve_agn_model()` | `get_agn_model = resolve_agn_model` |
| `models/dust/emission.py` | `get_emission_model()` | `resolve_emission_model()` | `get_emission_model = resolve_emission_model` |

**Strategy for each:**
1. Rename the function definition to the canonical name.
2. Add old name as module-level alias (simple assignment — no deprecation warning needed for internal-only functions; add `@deprecated_alias` if the old name is part of the public API).
3. Update all callers within `src/`.
4. Update all callers within `tests/`.

### 3.4 Verify

```bash
pytest tests/ -q
ruff check src/ tests/
```

---

## Phase 4: Parameter naming cleanup (deferred — higher risk)

This phase touches the translation layer and is more likely to break things. Execute only after Phases 1–3 are verified and merged.

### 4.1 Remove `_REVERSE_ALIASES` from `param_translate.py`

- Delete the `_REVERSE_ALIASES` dict.
- Delete the `find_legacy_param()` function.
- Update callers to use canonical prefixed names (Contract §3 prefix table).

### 4.2 Remove legacy `PARAM_MAP` from `param_translate.py`

- Delete the module-level `PARAM_MAP` dict.
- Ensure `_build_param_map()` is the sole translation mechanism.

### 4.3 Audit `resolve_short_names()` usage

- Confirm it is called only from `SEDModel.from_config()`.
- If called elsewhere, refactor those call sites to use fully-prefixed names.

### 4.4 Migrate bare `ValueError`/`RuntimeError` raises to Tengri exceptions (Contract §8)

Search:
```bash
grep -rn "raise ValueError\|raise RuntimeError" src/tengri/
```

Replace with `ParameterError`, `ConfigError`, `BackendError`, or `InferenceError` as appropriate. This is a behavioral change (callers catching `ValueError` will miss the new type), so:
- All new exceptions inherit from the original builtin via multiple inheritance if needed, OR
- Accept the break and document it in `CHANGELOG.md`.

Recommended: inherit. E.g., `class ParameterError(TengriError, ValueError)`.

### 4.5 Verify

```bash
pytest tests/ -q
ruff check src/ tests/
```

---

## Phase 5: SFH function canonical names (Contract §6)

### 5.1 Add canonical long names

**File:** `src/tengri/models/sfh/mean_sfh.py`

The short names (`tsnorm`, `snorm`, etc.) are the current function definitions. Add the canonical long names as the primary definitions and reassign:

```python
def truncated_skewnormal_sfh(...):
    """Truncated skew-normal SFH. Registry key: ``tsnorm``."""
    ...  # existing tsnorm body

# Registry key alias (not deprecated — kept permanently as SFH_REGISTRY keys)
tsnorm = truncated_skewnormal_sfh
```

Repeat for `skewnormal_sfh`/`snorm`, `lognormal_sfh`/`lnorm`, `gaussian_sfh`/`norm`, `double_powerlaw`/`dpl`.

### 5.2 Update `SFH_REGISTRY`

Ensure the registry maps **both** the short key and the canonical name to the same function:

```python
SFH_REGISTRY = {
    "tsnorm": truncated_skewnormal_sfh,
    "truncated_skewnormal_sfh": truncated_skewnormal_sfh,
    ...
}
```

### 5.3 Update `__init__.py` exports

Export both names. The short aliases are not deprecated (they serve as permanent registry keys per Contract §6).

### 5.4 Verify

```bash
pytest tests/ -q
ruff check src/ tests/
```

---

## Phase 6: Inference method alias wiring (Contract §5)

### 6.1 Centralize method string resolution

**File:** `src/tengri/inference/fitter.py` (or a dedicated `src/tengri/inference/method_registry.py`)

Ensure a single `resolve_method()` function handles all 13 canonical strings and all deprecated aliases from Contract §5. The function should:

1. Check if the string is canonical → return as-is.
2. Check if the string is a deprecated alias → emit `DeprecationWarning`, return the canonical string.
3. Otherwise → raise `ParameterError` (from Phase 0) with a message listing valid methods.

### 6.2 Audit existing dispatch code

Search for hardcoded method strings:
```bash
grep -rn "'geovi'\|'mgvi'\|'raytrace'\|'nuts'\|'nss'\|'elliptical_slice'" src/tengri/
```

Replace any direct comparisons with calls to `resolve_method()`.

### 6.3 Verify

```bash
pytest tests/ -q
```

---

## Subagent Execution Guide

Each phase is independent and can be assigned to a separate subagent. For each phase:

1. **Read the Naming Contract** at `docs/dev/NAMING_CONTRACT.md` first.
2. **Read this plan** for the specific phase being executed.
3. **Run the baseline** test suite and record the pass count before any changes.
4. **Search** for all usages of the old name before renaming:
   ```bash
   grep -rn "old_name" src/ tests/
   ```
5. **Rename** the class/function definition.
6. **Rename the file** if the Contract §2 table specifies a different module path than the current one. Do this in the same commit as the class rename.
7. **Add backward-compat alias** at module level. Use `deprecated_class_alias()` or `@deprecated_alias` from `utils/deprecation.py` for public API names. Use simple assignment for internal-only aliases.
8. **Update all internal callers** (within `src/`).
9. **Update all test files** (within `tests/`).
10. **Run verification:**
    ```bash
    pytest tests/ -q && ruff check src/ tests/ && ruff format --check src/ tests/
    ```
11. **Do NOT update** in this refactor: notebooks, examples, analysis scripts, user-facing docs — those are separate cleanup tasks tracked in `docs/dev/REFACTOR.md`.

### Ordering constraints

```
Phase 0  (no dependencies)
Phase 1  (depends on Phase 0 for exception classes if migrating errors simultaneously)
Phase 2  (independent of Phase 1; can run in parallel)
Phase 3  (independent of Phases 1–2; can run in parallel)
Phase 4  (depends on Phases 1–3 being merged)
Phase 5  (independent; can run anytime after Phase 0)
Phase 6  (depends on Phase 0 for ParameterError)
```

### Critical invariants

- **No logic changes.** This is a naming-only refactor. If a test fails, the rename was done incorrectly — revert and investigate.
- **No new public API surface.** The canonical names replace old names; they do not add new functionality.
- **One commit per sub-phase.** Each 2.x sub-phase is a separate commit for clean revert if needed.
- **File renames and class renames are atomic.** Never rename a file without updating all imports in the same commit.