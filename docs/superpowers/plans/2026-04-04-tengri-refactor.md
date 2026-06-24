# Tengri Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce structural debt in tengri by renaming mis-named types, splitting god files, fixing science correctness bugs, and adding gradient test infrastructure — without changing any public behavior.

**Architecture:** Seven independent subagent scopes (A–F can run in parallel in Wave 1; G depends on C completing first in Wave 2). Each scope is a self-contained refactor that leaves all 1764 tests passing. Zero behavior changes in Scopes A–C.

**Tech Stack:** JAX (jax, jax.numpy), Python 3.11+, pytest, ruff (linter/formatter), jupytext (notebooks).

---

## Dependency Graph

```
Wave 1 (all parallel):
  Scope A — Naming refactor
  Scope B — Split fitter.py
  Scope C — Split model.py
  Scope D — Science fix: CLOUDY bilinear (verify first — may already be done)
  Scope E — Science fix: marginalization ln_L (verify first — may already be done)
  Scope F — Gradient test infrastructure

Wave 2 (after C completes):
  Scope G — Tier 2 SED path

Wave 3 (standalone, low priority):
  Scope H — Prune stub documentation (Phase 7)
```

## Before You Start: Verification Checklist

Before implementing Scopes D and E, **verify the bugs still exist** — recent commits may have already fixed them:

```bash
# Scope D: does bilinear interpolation already use all 4 corners?
grep -n "_CLOUDY_SUBSOLAR_LOGU2\|z_frac.*u_frac" \
  src/tengri/models/observation/eline_priors.py
# If all 4 corners appear in the interpolation formula → Scope D is DONE. Write tests only.

# Scope E: does marginalize_emission_lines_cloudy shift the residual?
grep -n "residual_shifted\|prior_mean" \
  src/tengri/models/observation/eline_priors.py
# If residual shifting is present → the normalization fix is already applied. Write tests only.
```

---

## Scope A — Structural Naming Contract (Phase 1)

**Strategy:** Add new names with `DeprecationWarning` on old; update `__init__.py` exports; update all internal import sites, tests, and notebooks. Old names remain working until v1.0 removal.

**Files:**
- Modify: `src/tengri/core/param_spec.py`
- Modify: `src/tengri/inference/fitter.py`
- Modify: `src/tengri/inference/posterior.py`
- Modify: `src/tengri/inference/hierarchical.py`
- Modify: `src/tengri/models/observation/line_catalog.py`
- Modify: `src/tengri/__init__.py`
- Modify: `tests/` (update all import sites)
- Modify: `notebooks/` (update all import sites)
- Modify: `CLAUDE.md` (update class name references in the public API section)

### Task A-1: Add `Parameters` alias for `ParamSpec`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_naming_aliases.py
import warnings
from tengri import Parameters, ParamSpec
from tengri.distributions import Uniform

def test_parameters_alias_works():
    """Parameters is the new name for ParamSpec."""
    params = Parameters(
        sfh_tsnorm_log_total_mass=Uniform(-1, 2),
        redshift=0.1,
    )
    assert params.n_free >= 1

def test_paramspec_warns():
    """Old name emits DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        spec = ParamSpec(redshift=0.1)
        assert any("ParamSpec" in str(warning.message) for warning in w)
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ~/Projects/tengri && source .venv/bin/activate
pytest tests/unit/test_naming_aliases.py::test_parameters_alias_works -v
# Expected: FAIL — name 'Parameters' not in tengri
```

- [ ] **Step 3: Add `Parameters` alias in `param_spec.py`**

At the bottom of `src/tengri/core/param_spec.py`, add:

```python
# ---------------------------------------------------------------------------
# Naming alias — v1.0 will remove ParamSpec entirely
# ---------------------------------------------------------------------------

class _DeprecatedParamSpec(ParamSpec):
    """Deprecated alias for Parameters (formerly ParamSpec)."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

    def __new__(cls, *args, **kwargs):
        import warnings
        warnings.warn(
            "ParamSpec is deprecated. Use Parameters instead. "
            "ParamSpec will be removed in tengri v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__new__(cls)


# Rename the real class; keep old name as deprecated wrapper
Parameters = ParamSpec

def _make_paramspec_deprecated():
    """Return a ParamSpec subclass that warns on instantiation."""
    import warnings

    class _ParamSpecDeprecated(ParamSpec):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "ParamSpec is deprecated. Use Parameters instead. "
                "ParamSpec will be removed in tengri v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    _ParamSpecDeprecated.__name__ = "ParamSpec"
    _ParamSpecDeprecated.__qualname__ = "ParamSpec"
    return _ParamSpecDeprecated
```

**Note:** The cleanest pattern for a same-file deprecation is to rename the real class, keep the old name as a thin wrapper. The class is currently named `ParamSpec` — rename it to `Parameters` in-place, then add the deprecated `ParamSpec` wrapper below it:

1. In `param_spec.py`, change `class ParamSpec:` → `class Parameters:`
2. At the end of the file, add:

```python
def __init_deprecated_paramspec():
    import warnings

    class ParamSpec(Parameters):  # noqa: N801
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "ParamSpec is deprecated. Use Parameters instead. "
                "Will be removed in tengri v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    return ParamSpec


ParamSpec = __init_deprecated_paramspec()
```

- [ ] **Step 4: Update `__init__.py` exports**

In `src/tengri/__init__.py`, add `Parameters` to the export list:

```python
from tengri.protocols.param_spec import Parameters, ParamSpec  # ParamSpec deprecated
```

- [ ] **Step 5: Run tests to verify A-1 passes**

```bash
pytest tests/unit/test_naming_aliases.py -v
# Expected: PASS
ruff check src/ tests/
```

- [ ] **Step 6: Commit**

```bash
git add src/tengri/core/param_spec.py src/tengri/__init__.py tests/unit/test_naming_aliases.py
git commit -m "feat: add Parameters alias for ParamSpec (ParamSpec deprecated, removed v1.0)"
```

---

### Task A-2: Add `Spectroscopy` alias for `SpectroscopyConfig`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_naming_aliases.py`:

```python
import numpy as np
from tengri import Spectroscopy, SpectroscopyConfig

def test_spectroscopy_alias_works():
    spec = Spectroscopy(wave_obs=np.linspace(4000, 7000, 100))
    assert spec.wave_obs is not None

def test_spectroscopy_config_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SpectroscopyConfig(wave_obs=np.linspace(4000, 7000, 100))
        assert any("SpectroscopyConfig" in str(warning.message) for warning in w)
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_naming_aliases.py::test_spectroscopy_alias_works -v
# Expected: FAIL
```

- [ ] **Step 3: Apply rename in `src/tengri/models/observation/spectroscopy_config.py`**

Rename `class SpectroscopyConfig` → `class Spectroscopy`, then add deprecated alias at end of file:

```python
# Deprecated alias
def _make_deprecated_spectroscopy_config():
    import warnings

    class SpectroscopyConfig(Spectroscopy):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "SpectroscopyConfig is deprecated. Use Spectroscopy instead. "
                "Will be removed in tengri v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    return SpectroscopyConfig


SpectroscopyConfig = _make_deprecated_spectroscopy_config()
```

- [ ] **Step 4: Update `__init__.py`**

```python
from tengri.observation.spectroscopy_config import Spectroscopy, SpectroscopyConfig
```

- [ ] **Step 5: Run tests and ruff**

```bash
pytest tests/ -q -x
ruff check src/ tests/ && ruff format --check src/ tests/
```

- [ ] **Step 6: Commit**

```bash
git add src/tengri/models/observation/spectroscopy_config.py src/tengri/__init__.py tests/unit/test_naming_aliases.py
git commit -m "feat: add Spectroscopy alias for SpectroscopyConfig (deprecated)"
```

---

### Task A-3: Remaining renames (NoiseConfig→NoiseModel, HierarchicalFitter→PopulationFitter, HierarchicalResult→PopulationPosterior, LineCatalog→LineList)

Apply the same pattern as A-1/A-2 for each:

| File | Old class | New class |
|------|-----------|-----------|
| `models/observation/noise_config.py` | `NoiseConfig` | `NoiseModel` |
| `inference/hierarchical.py` | `HierarchicalFitter` | `PopulationFitter` |
| `inference/hierarchical.py` | `HierarchicalResult` | `PopulationPosterior` |
| `models/observation/line_catalog.py` | `LineCatalog` | `LineList` |

For each:
- [ ] **Write failing test** (follows pattern from A-1)
- [ ] **Rename real class, add deprecated alias at end of file**
- [ ] **Update `src/tengri/__init__.py`** — add new name, keep old with deprecation
- [ ] **Run `pytest tests/ -q` and `ruff check`**
- [ ] **Commit** with message `"feat: add [NewName] alias for [OldName] (deprecated)"`

For `LineCatalog → LineList`, also update `models/observation/line_catalog.py` filename? **No** — keep the filename. Only the class name changes. Renaming the file would break all imports.

---

### Task A-4: Rename `Posterior.summary()` → `Posterior.stats()`

This is slightly different — it's a method rename, not a class rename.

- [ ] **Write failing test**

```python
# tests/unit/test_naming_aliases.py
def test_posterior_stats_method_exists(mock_posterior):
    """stats() returns a dict; summary() is deprecated."""
    result = mock_posterior.stats()
    assert isinstance(result, dict)

def test_posterior_summary_warns(mock_posterior):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mock_posterior.summary()
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
```

Note: `mock_posterior` fixture needs to return a `Posterior` with mock chains. Add fixture to `conftest.py` or inline it using `Posterior.__new__` with mock attributes.

- [ ] **In `src/tengri/inference/posterior.py`:** rename the dict-returning method from `summary()` to `stats()`, then add:

```python
def summary(self, *args, **kwargs):
    """Deprecated. Use stats() instead."""
    import warnings
    warnings.warn(
        "Posterior.summary() is deprecated. Use Posterior.stats() instead. "
        "Will be removed in tengri v1.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return self.stats(*args, **kwargs)
```

- [ ] **Search for all callers of `.summary()` on Posterior objects**

```bash
grep -rn "\.summary()" tests/ notebooks/ src/ | grep -v "model\.\|spec\.\|fitter\.\|#"
```

Update any call sites that mean `Posterior.summary()` → `Posterior.stats()`.

- [ ] **Run `pytest tests/ -q`**
- [ ] **Commit**

```bash
git commit -m "feat: add Posterior.stats() method; Posterior.summary() deprecated"
```

---

### Task A-5: Update CLAUDE.md class name table

- [ ] In `CLAUDE.md`, update the High-level API section to show `Parameters`, `Spectroscopy`, `NoiseModel`, `PopulationFitter`, `LineList` as the canonical names, with a note that the old names still work but emit `DeprecationWarning`.

- [ ] **Final check: run full test suite**

```bash
pytest tests/ -q
# Expected: all pass (≈1764 tests)
ruff check src/ tests/ && ruff format --check src/ tests/
```

---

## Scope B — Split fitter.py (Phase 2A)

**Strategy:** Extract `_run_*` methods from `fitter.py` into module-level functions in new files. `fitter.py` becomes a dispatch table that imports and calls them. Zero behavior changes.

**Files:**
- Modify: `src/tengri/inference/fitter.py` (reduce from 4305 → ~150 lines of dispatch)
- Create: `src/tengri/inference/vi.py` (geoVI + MGVI logic)
- Create: `src/tengri/inference/mcmc.py` (NUTS + Ray Tracing + ESS)
- Create: `src/tengri/inference/evidence.py` (NSS)
- Create: `src/tengri/inference/map_dispatch.py` (MAP logic; `map.py` conflicts with Python built-in)
- Modify: `src/tengri/inference/__init__.py`

### Task B-1: Establish baseline test fingerprint

- [ ] **Record current test results for fitter-related tests**

```bash
cd ~/Projects/tengri && source .venv/bin/activate
pytest tests/ -q 2>&1 | tail -5 > /tmp/fitter_baseline.txt
cat /tmp/fitter_baseline.txt
# Record the exact "X passed, Y skipped" line for comparison after refactor
```

### Task B-2: Extract VI methods to `inference/vi.py`

The methods to extract: `_run_fast_vi` (line ~3227), `_run_native_vi` (line ~2442), `_run_nifty_vi` (line ~4042), and their private helpers.

- [ ] **Write a smoke test that will catch import or dispatch breakage**

```python
# tests/unit/test_fitter_split.py
def test_vi_module_importable():
    """After split, vi.py must be importable and expose run_fast_vi."""
    from tengri.inference.vi import run_fast_vi
    assert callable(run_fast_vi)

def test_mcmc_module_importable():
    from tengri.inference.mcmc import run_raytrace, run_nuts, run_elliptical_slice
    assert callable(run_raytrace)

def test_evidence_module_importable():
    from tengri.inference.evidence import run_nss
    assert callable(run_nss)

def test_map_dispatch_importable():
    from tengri.inference.map_dispatch import run_map
    assert callable(run_map)
```

- [ ] **Run test to verify it fails**

```bash
pytest tests/unit/test_fitter_split.py -v
# Expected: FAIL — modules don't exist yet
```

- [ ] **Create `src/tengri/inference/vi.py`**

Structure:
```python
"""Variational inference runners for tengri.

Extracted from fitter.py. Called by Fitter.run() dispatch table.
Each function takes (fitter_instance, *, key, **kwargs) and returns a Posterior.
"""
from __future__ import annotations
# ... imports ...

def run_fast_vi(fitter, *, key, **kwargs):
    """geoVI / MGVI fast path (NIFTy JIT). Extracted from Fitter._run_fast_vi."""
    # [paste body of _run_fast_vi here, replacing self → fitter]
    ...

def run_native_vi(fitter, *, key, **kwargs):
    """Native JAX geoVI path. Extracted from Fitter._run_native_vi."""
    ...

def run_nifty_vi(fitter, *, key, **kwargs):
    """NIFTy tight-loop geoVI with full logging. Extracted from Fitter._run_nifty_vi."""
    ...
```

**Extraction procedure:**
1. For each `_run_fast_vi`, `_run_native_vi`, `_run_nifty_vi` method in `fitter.py`:
   - Copy the method body into a module-level function `run_fast_vi(fitter, *, key, **kwargs):`
   - Replace every `self.` with `fitter.`
   - Ensure all helper functions used (e.g., `_simple_cg`) are either imported from fitter or moved to a `_vi_helpers.py`
2. In `fitter.py`, replace method bodies with one-liners:

```python
def _run_fast_vi(self, *, key, **kwargs):
    from tengri.inference.vi import run_fast_vi
    return run_fast_vi(self, key=key, **kwargs)
```

- [ ] **Create `src/tengri/inference/mcmc.py`** (same pattern for `_run_raytrace`, `_run_nuts`, `_run_elliptical_slice`)

```python
"""MCMC runners: Ray Tracing, NUTS, and Elliptical Slice Sampling."""
from __future__ import annotations

def run_raytrace(fitter, *, key, **kwargs): ...
def run_nuts(fitter, *, key, **kwargs): ...
def run_elliptical_slice(fitter, *, key, **kwargs): ...
```

- [ ] **Create `src/tengri/inference/evidence.py`** (for `_run_nss`)

```python
"""Nested Slice Sampling for Bayesian evidence (log Z)."""
from __future__ import annotations

def run_nss(fitter, *, key, **kwargs): ...
```

- [ ] **Create `src/tengri/inference/map_dispatch.py`** (for `_run_map`, `_run_laplace`, `_run_pathfinder`)

```python
"""MAP optimization, Laplace approximation, and Pathfinder."""
from __future__ import annotations

def run_map(fitter, *, key, **kwargs): ...
def run_laplace(fitter, *, key, **kwargs): ...
def run_pathfinder(fitter, *, key, **kwargs): ...
```

- [ ] **Run smoke tests**

```bash
pytest tests/unit/test_fitter_split.py -v
# Expected: PASS
```

### Task B-3: Verify full test suite still passes

- [ ] **Run full suite**

```bash
pytest tests/ -q
# Compare output to /tmp/fitter_baseline.txt — must match exactly
```

- [ ] **Check line count of fitter.py**

```bash
wc -l src/tengri/inference/fitter.py
# Target: < 500 lines after extraction (dispatch table + class setup)
# The run() method + dispatch branches + _DEPRECATED_METHOD_ALIASES + Fitter.__init__ should be ~300 lines
```

- [ ] **Run ruff**

```bash
ruff check src/tengri/inference/ && ruff format --check src/tengri/inference/
```

- [ ] **Commit**

```bash
git add src/tengri/inference/vi.py src/tengri/inference/mcmc.py \
    src/tengri/inference/evidence.py src/tengri/inference/map_dispatch.py \
    src/tengri/inference/fitter.py tests/unit/test_fitter_split.py
git commit -m "refactor: split fitter.py into vi.py, mcmc.py, evidence.py, map_dispatch.py"
```

---

## Scope C — Split model.py (Phase 2B)

**Strategy:** Extract convenience methods and mock methods from `model.py` into `core/convenience.py`. `model.py` stays as the orchestrator (thin `__init__`, `from_config`, `fit`, `predict_*`).

**Files:**
- Modify: `src/tengri/core/model.py` (reduce from 2615 → ~600 lines)
- Create: `src/tengri/core/convenience.py`
- Modify: `src/tengri/core/__init__.py`

**Methods to extract** (all are currently on `Model`):
- `prior_predictive` (line ~2033)
- `fit_catalog` (line ~2182)
- `fit_population` (line ~2534)
- `mock`, `mock_spectrum`, `mock_batch` (lines ~1776–1853)
- `predict_photometry_batch`, `predict_spectrum_batch` (lines ~1853–1887)

### Task C-1: Write baseline + smoke tests

- [ ] **Write smoke tests**

```python
# tests/unit/test_model_split.py
def test_convenience_module_importable():
    from tengri.protocols.convenience import (
        prior_predictive,
        fit_catalog,
        fit_population,
    )
    assert callable(prior_predictive)

def test_model_still_has_convenience_methods(mock_model):
    """Model.prior_predictive must still work as a method."""
    # prior_predictive is delegated, not removed
    assert hasattr(mock_model, "prior_predictive")
    assert callable(mock_model.prior_predictive)
```

- [ ] **Run to verify they fail**

```bash
pytest tests/unit/test_model_split.py -v
# Expected: FAIL
```

### Task C-2: Create `core/convenience.py`

```python
"""Convenience wrappers delegated from Model.

These are extracted from core/model.py to keep model.py focused on
the forward model (predict_* methods and __init__).
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tengri.forward.sed_model import Model


def prior_predictive(model: "Model", n: int = 500, seed: int = 42):
    """Prior predictive check. See Model.prior_predictive for full docs."""
    # [paste body of Model.prior_predictive here, replacing self → model]
    ...


def fit_catalog(model: "Model", catalog, **kwargs):
    """Batch catalog fitting. See Model.fit_catalog for full docs."""
    ...


def fit_population(model: "Model", obs_list, **kwargs):
    """Population/hierarchical fitting. See Model.fit_population for full docs."""
    ...
```

**Extraction procedure** (same as Scope B):
1. Paste each method body into a module-level function `def func(model, ...):`
2. Replace `self.` with `model.`
3. Replace `self` alone with `model`
4. In `model.py`, replace each method with a one-liner:

```python
def prior_predictive(self, n: int = 500, seed: int = 42):
    from tengri.protocols.convenience import prior_predictive as _pp
    return _pp(self, n=n, seed=seed)
```

### Task C-3: Verify and commit

- [ ] **Run full suite**

```bash
pytest tests/ -q
# Must match baseline
```

- [ ] **Check model.py size**

```bash
wc -l src/tengri/core/model.py
# Target: < 1800 lines (extracting ~800 lines of convenience methods)
```

- [ ] **Run ruff**

```bash
ruff check src/tengri/core/ && ruff format --check src/tengri/core/
```

- [ ] **Commit**

```bash
git add src/tengri/core/convenience.py src/tengri/core/model.py tests/unit/test_model_split.py
git commit -m "refactor: extract convenience methods from model.py into core/convenience.py"
```

---

## Scope D — Science Fix: CLOUDY 2D Interpolation (Phase 6A)

**⚠️ Verify first** — run the check from the top of this document. If the 4-corner bilinear interpolation is already present, skip directly to Task D-2 (write regression tests only).

### Task D-1: Verify bug status

- [ ] **Run the regression test from REFACTOR.md**

```bash
python -c "
import jax.numpy as jnp
from tengri.observation.eline_priors import cloudy_line_priors
means_solar, _ = cloudy_line_priors(log_z=0.0, neb_logU=-2.0)
means_subsolar, _ = cloudy_line_priors(log_z=-0.7, neb_logU=-2.0)
# [NII]6583 is index 8 in the CLOUDY reference line list
print('Solar [NII]6583:', float(means_solar[8]))
print('Subsolar [NII]6583:', float(means_subsolar[8]))
print('Ratio:', float(means_subsolar[8] / means_solar[8]))
print('Bug present (>0.5)?', float(means_subsolar[8] / means_solar[8]) >= 0.5)
"
# If ratio < 0.5 → bug is already fixed. Skip to Task D-2.
# If ratio >= 0.5 → bug still present, apply fix below.
```

**If bug is present:** In `src/tengri/models/observation/eline_priors.py`, ensure all 4 grid corners are used:

```python
# Ensure _CLOUDY_SUBSOLAR_LOGU2 constant is defined (sub-solar Z, logU=-2)
# Then the bilinear interpolation must be:
prior_means = (
    (1.0 - z_frac) * (1.0 - u_frac) * _CLOUDY_SUBSOLAR_LOGU3
    + z_frac * (1.0 - u_frac) * _CLOUDY_SOLAR_LOGU3
    + (1.0 - z_frac) * u_frac * _CLOUDY_SUBSOLAR_LOGU2
    + z_frac * u_frac * _CLOUDY_SOLAR_LOGU2
)
```

Values for `_CLOUDY_SUBSOLAR_LOGU2` (derived from Byler+2017 CLOUDY trends, sub-solar Z, logU=-2):
```python
_CLOUDY_SUBSOLAR_LOGU2 = jnp.array([
    0.25,  # [OII] 3726
    0.32,  # [OII] 3729
    0.26,  # H-delta
    0.47,  # H-gamma
    1.00,  # H-beta
    1.80,  # [OIII] 4959
    5.40,  # [OIII] 5007
    0.02,  # [NII] 6548
    2.86,  # H-alpha
    0.05,  # [NII] 6583
    0.05,  # [SII] 6716
    0.04,  # [SII] 6731
])
```

### Task D-2: Write regression tests

- [ ] **Write tests** (these are required regardless of whether the bug was pre-fixed)

```python
# tests/unit/test_eline_priors.py (new file or add to existing)
import jax.numpy as jnp
from tengri.observation.eline_priors import cloudy_line_priors


def test_cloudy_priors_metallicity_effect_at_high_logu():
    """[NII]6583 must be weaker at sub-solar Z vs solar Z at logU=-2 (NEW-01 regression)."""
    means_solar, _ = cloudy_line_priors(log_z=0.0, neb_logU=-2.0)
    means_subsolar, _ = cloudy_line_priors(log_z=-0.7, neb_logU=-2.0)
    # [NII]6583 is index 8 in the CLOUDY reference 12-line list
    assert float(means_subsolar[8]) < 0.5 * float(means_solar[8]), (
        f"[NII]6583 at sub-solar Z ({means_subsolar[8]:.3f}) should be <50% of "
        f"solar Z value ({means_solar[8]:.3f}) at logU=-2"
    )


def test_cloudy_priors_solar_logu3_corner():
    """At solar Z + logU=-3, result matches the SOLAR_LOGU3 reference."""
    from tengri.observation.eline_priors import _CLOUDY_SOLAR_LOGU3
    means, _ = cloudy_line_priors(log_z=0.0, neb_logU=-3.0)
    assert jnp.allclose(means, _CLOUDY_SOLAR_LOGU3, atol=1e-5)


def test_cloudy_priors_bilinear_varies_with_both_axes():
    """Result must vary when either log_z OR neb_logU changes."""
    base, _ = cloudy_line_priors(log_z=0.0, neb_logU=-3.0)
    z_varied, _ = cloudy_line_priors(log_z=-0.35, neb_logU=-3.0)
    u_varied, _ = cloudy_line_priors(log_z=0.0, neb_logU=-2.5)
    assert not jnp.allclose(base, z_varied), "Result must vary with log_z"
    assert not jnp.allclose(base, u_varied), "Result must vary with neb_logU"


def test_cloudy_priors_gradient_is_finite():
    """JAX gradient through cloudy_line_priors must be finite (JIT-safe)."""
    import jax
    grad_fn = jax.grad(lambda z: cloudy_line_priors(log_z=z, neb_logU=-2.5)[0].sum())
    g = grad_fn(0.0)
    assert jnp.isfinite(g), f"Gradient is not finite: {g}"
```

- [ ] **Run tests**

```bash
pytest tests/unit/test_eline_priors.py -v
# Expected: all PASS
```

- [ ] **Commit**

```bash
git add tests/unit/test_eline_priors.py src/tengri/models/observation/eline_priors.py
git commit -m "fix: CLOUDY bilinear interpolation regression tests (NEW-01)"
# Or if fix was applied: "fix: CLOUDY 4-corner bilinear interpolation for cloudy_line_priors (NEW-01)"
```

---

## Scope E — Science Fix: Marginalization ln_L (Phase 6B)

**⚠️ Verify first** — check whether `marginalize_emission_lines_cloudy` already uses residual shifting (which IS mathematically equivalent to the explicit normalization term).

### Task E-1: Verify bug status

- [ ] **Run verification**

```bash
python -c "
import inspect
from tengri.observation.eline_priors import marginalize_emission_lines_cloudy
src = inspect.getsource(marginalize_emission_lines_cloudy)
print('residual_shifted present:', 'residual_shifted' in src)
print('prior_mean pre-subtracted:', 'design_matrix @ scaled_means' in src)
"
# If both True → residual shifting is implemented → normalization bug likely fixed.
# Verify correctness with the finite-difference test below.
```

- [ ] **Finite-difference check of the gradient**

```python
# Run this in a Python REPL to verify gradient correctness
import jax
import jax.numpy as jnp
from tengri.observation.eline_priors import marginalize_emission_lines_cloudy

key = jax.random.PRNGKey(0)
n_pix, n_lines = 50, 3
residual = jax.random.normal(key, (n_pix,))
noise = jnp.ones(n_pix) * 0.1
design = jax.random.normal(jax.random.PRNGKey(1), (n_pix, n_lines))

def ln_l_fn(log_z):
    ln_l, _, _ = marginalize_emission_lines_cloudy(
        residual, noise, design, log_z=log_z, neb_logU=-2.5
    )
    return ln_l

# Analytic gradient
g_analytic = jax.grad(ln_l_fn)(0.0)

# Finite-difference gradient
eps = 1e-4
g_fd = (ln_l_fn(eps) - ln_l_fn(-eps)) / (2 * eps)
print(f"Analytic: {g_analytic:.6f}, FD: {g_fd:.6f}, match: {abs(g_analytic - g_fd) < 1e-3}")
```

If they match → normalization is correct. If not → apply the explicit normalization fix.

**If fix is needed:** Add the `+0.5 × μ²/σ²` term explicitly in `marginalize_emission_lines_cloudy`:

```python
# After computing ln_l_marg from the zero-mean marginalization:
# Add the prior normalization term for the non-zero mean
# This accounts for the shift in the Gaussian prior:
# ln p(a=0; mu, sigma^2) - ln p(a=0; 0, sigma^2) = -0.5 * mu^2 / sigma^2
prior_norm_correction = 0.5 * jnp.sum(scaled_means**2 / scaled_sigmas**2)
ln_l_marg = ln_l_marg + prior_norm_correction
```

### Task E-2: Write regression tests

- [ ] **Write tests**

```python
# tests/unit/test_eline_priors.py (add to existing)
import jax
import jax.numpy as jnp


def test_marginalize_gradient_is_finite():
    """Gradient of marginalized ln_L wrt log_z must be finite."""
    from tengri.observation.eline_priors import marginalize_emission_lines_cloudy

    n_pix, n_lines = 50, 3
    residual = jnp.zeros(n_pix)
    noise = jnp.ones(n_pix) * 0.1
    design = jnp.ones((n_pix, n_lines)) * 0.01

    def ln_l_fn(log_z):
        ln_l, _, _ = marginalize_emission_lines_cloudy(
            residual, noise, design, log_z=log_z, neb_logU=-2.5
        )
        return ln_l

    g = jax.grad(ln_l_fn)(0.0)
    assert jnp.isfinite(g), f"Gradient not finite: {g}"


def test_marginalize_gradient_matches_finite_difference():
    """Analytic gradient must match finite-difference to 0.1% (NEW-09 regression)."""
    from tengri.observation.eline_priors import marginalize_emission_lines_cloudy

    key = jax.random.PRNGKey(42)
    n_pix, n_lines = 30, 4
    residual = jax.random.normal(key, (n_pix,)) * 0.05
    noise = jnp.ones(n_pix) * 0.1
    design = jax.random.normal(jax.random.PRNGKey(1), (n_pix, n_lines)) * 0.01

    def ln_l_fn(log_z):
        ln_l, _, _ = marginalize_emission_lines_cloudy(
            residual, noise, design, log_z=log_z, neb_logU=-2.5
        )
        return ln_l

    eps = 1e-4
    g_analytic = float(jax.grad(ln_l_fn)(0.0))
    g_fd = float((ln_l_fn(eps) - ln_l_fn(-eps)) / (2 * eps))

    rel_err = abs(g_analytic - g_fd) / (abs(g_fd) + 1e-10)
    assert rel_err < 0.001, (
        f"Gradient mismatch: analytic={g_analytic:.6f}, FD={g_fd:.6f}, "
        f"relative error={rel_err:.4f}"
    )
```

- [ ] **Run tests**

```bash
pytest tests/unit/test_eline_priors.py -v
# Expected: all PASS
```

- [ ] **Commit**

```bash
git add tests/unit/test_eline_priors.py
git commit -m "test: finite-difference gradient regression for marginalize_emission_lines_cloudy (NEW-09)"
```

---

## Scope F — Gradient Test Infrastructure (Phase 6C)

**Goal:** Create `tests/unit/test_gradients.py` covering CSP weights, dust attenuation, nebular marginalization, IGM transmission, and AGN disc spectrum. All tests run without SSP data; mock inputs only. Full suite must complete in < 30 s.

### Task F-1: Write CSP mass weight gradient test

- [ ] **Write test**

```python
# tests/unit/test_gradients.py
"""Finite-difference gradient checks for all JIT boundary transforms.

These tests catch sign errors, missing terms, and non-differentiable branches
that isfinite() checks would miss. Each test uses jax.grad + manual FD.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)
_EPS = 1e-4
_REL_TOL = 0.005  # 0.5% relative tolerance for FD vs analytic


def _check_grad(fn, x0, eps=_EPS, tol=_REL_TOL):
    """Assert analytic gradient matches finite-difference at x0 (scalar input)."""
    g_analytic = float(jax.grad(fn)(x0))
    g_fd = float((fn(x0 + eps) - fn(x0 - eps)) / (2.0 * eps))
    rel_err = abs(g_analytic - g_fd) / (abs(g_fd) + 1e-12)
    assert rel_err < tol, (
        f"Gradient mismatch: analytic={g_analytic:.6f}, FD={g_fd:.6f}, "
        f"rel_err={rel_err:.4f} (tol={tol})"
    )


def test_sfh_transform_gradient():
    """SFH parametric transforms must have correct gradients."""
    from tengri.components.stellar.sfh.mean_sfh import tsnorm_sfh

    ages_gyr = jnp.linspace(0.01, 13.7, 200)

    def sfr_total(log_total_mass):
        sfr = tsnorm_sfh(
            ages_gyr,
            log_total_mass=log_total_mass,
            peak_lbt_gyr=5.0,
            width_gyr=2.0,
            skew=0.0,
            trunc=10.0,
        )
        return jnp.sum(sfr)

    _check_grad(sfr_total, 0.5)
    _check_grad(sfr_total, -0.5)
```

- [ ] **Run to verify it works**

```bash
pytest tests/unit/test_gradients.py::test_sfh_transform_gradient -v
# Expected: PASS
```

### Task F-2: Write dust attenuation gradient test

- [ ] **Add to `test_gradients.py`**

```python
def test_dust_attenuation_gradient():
    """Two-component Charlot-Fall dust must have correct gradient wrt tau_bc."""
    from tengri.components.dust.attenuation import two_component_dust

    wave_rest = jnp.linspace(1000.0, 10000.0, 300)
    sed = jnp.ones(300)

    def total_flux(tau_bc):
        tau_diff = 0.3
        attenuated = two_component_dust(
            wave_rest, sed,
            tau_bc=tau_bc,
            tau_diff=tau_diff,
            dust_slope=-0.7,
            age_weights=jnp.array([0.3, 0.4, 0.3]),
            age_grid_yr=jnp.array([1e7, 5e8, 5e9]),
        )
        return jnp.sum(attenuated)

    _check_grad(total_flux, 0.5)
    _check_grad(total_flux, 2.0)
```

**Note:** Adjust the `two_component_dust` call signature to match the actual API. Run `help(two_component_dust)` to confirm parameter names before writing the test.

### Task F-3: Write IGM transmission gradient test

- [ ] **Add to `test_gradients.py`**

```python
def test_igm_transmission_gradient():
    """IGM Inoue+2014 transmission must have finite gradient wrt redshift."""
    from tengri.components.igm import igm_transmission

    wave_obs = jnp.linspace(800.0, 3000.0, 100)  # observed-frame, Angstrom

    def total_transmission(z):
        trans = igm_transmission(wave_obs, z)
        return jnp.sum(trans)

    # Test at z=2.0 and z=3.5 where IGM absorption is significant
    _check_grad(total_transmission, 2.0)
    _check_grad(total_transmission, 3.5)
```

### Task F-4: Write nebular marginalization gradient test

- [ ] **Add to `test_gradients.py`** (reuse logic from Scope E)

```python
def test_nebular_marginalization_gradient():
    """Emission line marginalization ln_L must have correct gradient wrt log_z."""
    from tengri.observation.eline_priors import marginalize_emission_lines_cloudy

    n_pix, n_lines = 40, 5
    key = jax.random.PRNGKey(7)
    residual = jax.random.normal(key, (n_pix,)) * 0.05
    noise = jnp.ones(n_pix) * 0.1
    design = jax.random.normal(jax.random.PRNGKey(8), (n_pix, n_lines)) * 0.01

    def ln_l(log_z):
        ln_l_marg, _, _ = marginalize_emission_lines_cloudy(
            residual, noise, design, log_z=log_z, neb_logU=-2.5
        )
        return ln_l_marg

    _check_grad(ln_l, 0.0)
    _check_grad(ln_l, -0.5)
```

### Task F-5: Write AGN disc gradient test

- [ ] **Add to `test_gradients.py`**

```python
def test_agn_disc_gradient():
    """AGN multicolor disc SED must have finite gradient wrt log_mbh."""
    from tengri.components.agn.disc import multicolor_disc

    wave_rest = jnp.linspace(100.0, 30000.0, 200)

    def total_flux(log_mbh):
        sed = multicolor_disc(wave_rest, log_mbh=log_mbh, log_mdot=-1.0, spin=0.0)
        return jnp.sum(sed)

    _check_grad(total_flux, 8.0)
    _check_grad(total_flux, 9.5)
```

**Note:** Adjust function name and parameter names to match actual API. Use `grep -n "def multicolor" src/tengri/models/agn/disc.py` to confirm.

### Task F-6: Run full gradient test suite and check timing

- [ ] **Run all gradient tests with timing**

```bash
pytest tests/unit/test_gradients.py -v --tb=short 2>&1 | tee /tmp/gradient_tests.txt
# Check: all tests PASS
# Check: total time < 30 seconds
```

- [ ] **Run full suite to confirm no regressions**

```bash
pytest tests/ -q
ruff check src/ tests/ && ruff format --check src/ tests/
```

- [ ] **Commit**

```bash
git add tests/unit/test_gradients.py
git commit -m "test: gradient correctness suite for dust, SFH, IGM, nebular, AGN (Phase 6C)"
```

---

## Scope G — Tier 2 SED Path (Phase 5) — **Depends on Scope C completing first**

**Goal:** Ensure the compositional rest-frame SED path (`_compute_rest_sed_tier2`) is wired into the dispatch tier logic in `model.py`. Add benchmark test asserting free-z forward pass < 600 µs on CPU.

**Pre-requisite:** Check if tier-2 is already partially implemented.

```bash
grep -n "_compute_rest_sed_tier2\|_predict_photometry_tier2\|build_fused_rest_sed" \
  src/tengri/core/model.py src/tengri/core/fused_kernels.py
```

### Task G-1: Audit current tier-2 status

- [ ] **Check tier dispatch logic**

```bash
grep -n "tier\|tier1\|tier2\|tier3\|precomputed\|_fast\|_tier" src/tengri/core/model.py | head -30
```

Expected: `predict_photometry` routes to `_predict_photometry_fast` (tier 1) or falls through to a slower path. Tier 2 (`_compute_rest_sed_tier2`) exists but may not be wired into the dispatch.

- [ ] **Identify what `_compute_rest_sed_tier2` does vs. what it should do**

```bash
grep -n -A 5 "def _compute_rest_sed_tier2" src/tengri/core/model.py
```

### Task G-2: Write benchmark test

- [ ] **Write failing benchmark test**

```python
# tests/unit/test_tier2_benchmark.py
import time
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
_CPU_THRESHOLD_US = 600  # 600 µs for free-z forward pass

@pytest.mark.skipif(
    not _has_ssp_data(),  # helper to check for data/ssp_*.h5
    reason="SSP data not available"
)
def test_tier2_forward_pass_speed(mock_model_free_z):
    """Free-z forward model must run in < 600 µs after JIT warmup."""
    params = mock_model_free_z.spec.sample(jax.random.PRNGKey(0))

    # JIT warmup (compile)
    _ = mock_model_free_z.predict_photometry(params)

    # Benchmark: median of 20 calls
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        _ = mock_model_free_z.predict_photometry(params)
        jax.block_until_ready(_)
        times.append((time.perf_counter() - t0) * 1e6)

    median_us = sorted(times)[len(times) // 2]
    assert median_us < _CPU_THRESHOLD_US, (
        f"Tier-2 forward pass took {median_us:.0f} µs, expected < {_CPU_THRESHOLD_US} µs"
    )
```

### Task G-3: Extract component subfunctions from `sed_pipeline.py`

Per REFACTOR.md, extract into standalone pure functions in `core/sed_pipeline.py`:

```python
def _compute_stellar_sed(params, ssp, sfh_weights): ...
def _apply_dust_attenuation(sed, params, config): ...
def _add_dust_emission(sed, params, config): ...
def _add_agn(sed, params, config): ...
def _add_nebular(sed, params, config, nebular_backend): ...
```

These are called by `build_fused_rest_sed` in `fused_kernels.py`.

### Task G-4: Wire tier-2 dispatch in `predict_photometry`

In `model.py`, the dispatch order should be:
1. Tier 1: `if self._photometry_precomputed and self._z_fixed:` → `_predict_photometry_fast`
2. Tier 2: `elif self._tier2_available:` → `_predict_photometry_tier2`
3. Tier 3 (exact fallback): `else:` → `predict_sed` → integrate

- [ ] **Run benchmark test**

```bash
pytest tests/unit/test_tier2_benchmark.py -v -s
# Expected: PASS with timing < 600 µs
```

- [ ] **Run full suite**

```bash
pytest tests/ -q
```

- [ ] **Commit**

```bash
git add src/tengri/core/fused_kernels.py src/tengri/core/sed_pipeline.py \
    src/tengri/core/model.py tests/unit/test_tier2_benchmark.py
git commit -m "feat: wire tier-2 compositional SED dispatch in predict_photometry (Phase 5)"
```

---

## Scope H — Prune Stub Documentation (Phase 7)

**Goal:** Move zero-code model documentation out of `docs/models/` into a `ROADMAP.md` with 1–3 sentence motivation blurbs.

### Task H-1: Create ROADMAP.md

- [ ] **Create `ROADMAP.md` in project root**

```markdown
# Tengri Roadmap

## Planned Physics Modules

These modules are planned but not yet implemented. Do not document them in `docs/models/`.

### Chemical Evolution Z(t)
Time-evolving metallicity Z(t) coupled to the star formation history.
Motivation: breaks the age-metallicity degeneracy in old stellar populations.

### Shock Emission (MAPPINGS)
Radiative shock emission from MAPPINGS V grids.
Motivation: critical for AGN-host and starburst-driven outflow diagnostics.

### ADAF Disc
Advection-dominated accretion flow model for low-luminosity AGN (Mahadevan 1997).
Motivation: needed for quiescent black holes below the AGN threshold.

### MAGPHYS-style Dust
Energy-balance dust model from da Cunha+2008.
Motivation: alternative to DL07 for high-z submillimeter-selected sources.

### THEMIS Dust
Jones+2017 dust grain size distribution with amorphous carbon.
Motivation: physically motivated alternative to silicate/graphite mixtures.

### Patchy IGM Reionization
Neutral hydrogen bubble attenuation at z > 6 (two free parameters: x_HI, R_bubble).
Motivation: required for Lyman-alpha transmission statistics at cosmic dawn.

### PAH Emission Features
Mid-infrared PAH complex (6.2, 7.7, 8.6, 11.3 µm).
Motivation: strong star formation diagnostic for JWST/MIRI observations.
```

### Task H-2: Delete or gut stub model docs

For each stub doc listed in REFACTOR.md:
- `docs/models/chemical_evolution.md`
- `docs/models/shock_emission.md`
- `docs/models/adaf_disc.md`
- `docs/models/magphys_dust.md`
- `docs/models/themis_dust.md`
- `docs/models/patchy_igm.md`
- `docs/models/pah_features.md`

- [ ] **For each doc:** Delete the implementation sections (code examples, parameter tables). Keep the title and a 1-line reference to `ROADMAP.md`:

```markdown
# [Model Name]

See [ROADMAP.md](../../ROADMAP.md) for planned implementation notes.
```

Or delete the file entirely if it contains nothing beyond implementation stubs.

- [ ] **Commit**

```bash
git add ROADMAP.md docs/models/
git commit -m "docs: move unimplemented model stubs to ROADMAP.md (Phase 7)"
```

---

## Execution Summary

| Scope | Phase | Priority | Depends On | Expected Effort |
|-------|-------|----------|-----------|----------------|
| A | 1 | P0, non-breaking | — | Medium (grep+rename) |
| B | 2A | P0 | — | Large (4305→~150 line file) |
| C | 2B | P0 | — | Medium (2615→~800 line file) |
| D | 6A | P0 | — | Small (verify then test) |
| E | 6B | P0 | — | Small (verify then test) |
| F | 6C | P0 | — | Medium (new test file) |
| G | 5 | P1 | C | Large (tier dispatch + benchmark) |
| H | 7 | P2 | — | Small (doc edits) |

**Wave 1 (parallel):** A, B, C, D, E, F
**Wave 2 (after C):** G
**Wave 3 (any time):** H

**Phase 3 (Settings/Parameters split) is intentionally NOT included** — it is a breaking API change that restructures `ParamSpec.__init__` and should be a separate, carefully planned effort after the Paper I submission freeze.

---

## Self-Review Against Spec

Spec coverage check:

| REFACTOR.md requirement | Covered by |
|------------------------|-----------|
| ParamSpec → Parameters | Scope A, Task A-1 |
| SpectroscopyConfig → Spectroscopy | Scope A, Task A-2 |
| NoiseConfig → NoiseModel | Scope A, Task A-3 |
| HierarchicalFitter → PopulationFitter | Scope A, Task A-3 |
| HierarchicalResult → PopulationPosterior | Scope A, Task A-3 |
| LineCatalog → LineList | Scope A, Task A-3 |
| Posterior.summary() → Posterior.stats() | Scope A, Task A-4 |
| Split fitter.py | Scope B |
| Split model.py | Scope C |
| CLOUDY 2D interpolation | Scope D |
| Marginalization ln_L | Scope E |
| Gradient test infrastructure | Scope F |
| Tier 2 SED path | Scope G |
| Prune stub docs | Scope H |

**Not covered (intentional deferrals):**
- Phase 3 (Settings/Parameters split) — breaking change, deferred post Paper I
- BUG-29 (`_mstar` surviving mass) — needs DSPS output, not blocking Paper I
- `cloudy_grid_line_priors()` zero coverage (NEW-07) — extend Scope F to add this if time permits
