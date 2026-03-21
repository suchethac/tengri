# Package Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the flat diffsed package into `core/` (forward model) and expanded `inference/` (fitting) subpackages, reducing root clutter from 18 to 3 files.

**Architecture:** Move files into subpackages, update all internal imports (strategy 1: clean break, no shims), keep public API unchanged via `__init__.py` re-exports. Subpackage `__init__.py` files also export key classes for convenience (`from diffsed.core import Model`).

**Tech Stack:** Python, JAX, ruff, pytest

---

## Target Structure

```
src/diffsed/
├── __init__.py              # public API re-exports (unchanged interface)
├── distributions.py         # cross-cutting priors
├── plotting.py              # visualization
├── simulate.py              # SED-from-SFH utilities
│
├── core/                    # forward model
│   ├── __init__.py          # re-exports: Model, ParamSpec, Prediction, ...
│   ├── model.py             # Model class
│   ├── param_spec.py        # ParamSpec class
│   ├── param_translate.py   # param mapping + unit conversion
│   ├── fused_kernels.py     # JIT kernel builders
│   ├── sed_pipeline.py      # SED computation engine
│   ├── prediction.py        # lazy Prediction object
│   ├── noise.py             # noise model
│   └── mock.py              # generate_mock, MockData
│
├── inference/               # all fitting + results
│   ├── __init__.py          # re-exports: Fitter, Posterior, HierarchicalFitter, ...
│   ├── fitter.py            # Fitter class
│   ├── hierarchical.py      # HierarchicalFitter
│   ├── posterior.py          # Posterior results
│   ├── raytrace.py          # Ray Tracing sampler (was raytrace_jax.py)
│   ├── vi_config.py         # VI settings
│   ├── standardized.py      # standardized param handling
│   ├── common.py            # (already here)
│   ├── nuts.py              # (already here)
│   ├── geovi.py             # (already here)
│   ├── geovi_nuts.py        # (already here)
│   └── map_optimizer.py     # (already here)
│
├── models/                  # physics (unchanged)
├── utils/                   # helpers (unchanged)
└── diagnostics/             # analysis (unchanged)
```

## Import Mapping

| Old path | New path |
|----------|----------|
| `diffsed.model` | `diffsed.core.model` |
| `diffsed.param_spec` | `diffsed.core.param_spec` |
| `diffsed._param_translate` | `diffsed.core.param_translate` |
| `diffsed._fused_kernels` | `diffsed.core.fused_kernels` |
| `diffsed._sed_pipeline` | `diffsed.core.sed_pipeline` |
| `diffsed._mock` | `diffsed.core.mock` |
| `diffsed.prediction` | `diffsed.core.prediction` |
| `diffsed.noise` | `diffsed.core.noise` |
| `diffsed.fitter` | `diffsed.inference.fitter` |
| `diffsed.hierarchical` | `diffsed.inference.hierarchical` |
| `diffsed.posterior` | `diffsed.inference.posterior` |
| `diffsed.raytrace_jax` | `diffsed.inference.raytrace` |
| `diffsed.vi_config` | `diffsed.inference.vi_config` |
| `diffsed.standardized` | `diffsed.inference.standardized` |

---

## Task 1: Create `core/` subpackage and move files

**Files:**
- Create: `src/diffsed/core/__init__.py`
- Move: `model.py`, `param_spec.py`, `_param_translate.py`, `_fused_kernels.py`, `_sed_pipeline.py`, `_mock.py`, `prediction.py`, `noise.py` → `core/`
- Rename: drop `_` prefix on moved files

- [ ] **Step 1: Create core/ directory and __init__.py**

```bash
mkdir -p src/diffsed/core
```

Create `src/diffsed/core/__init__.py`:
```python
"""Core forward model: Model, ParamSpec, Prediction, and internals."""

from diffsed.core.mock import MockData, generate_mock
from diffsed.core.model import Model
from diffsed.core.noise import (
    compute_effective_noise,
    compute_std_inv,
    has_noise_model,
    uses_student_t,
    variable_noise_hamiltonian,
)
from diffsed.core.param_spec import ParamSpec
from diffsed.core.param_translate import LOG10_ZSUN
from diffsed.core.prediction import (
    DerivedQuantities,
    EmissionLines,
    Prediction,
    SEDQuantities,
    SFHQuantities,
)

__all__ = [
    "LOG10_ZSUN",
    "DerivedQuantities",
    "EmissionLines",
    "MockData",
    "Model",
    "ParamSpec",
    "Prediction",
    "SEDQuantities",
    "SFHQuantities",
    "compute_effective_noise",
    "compute_std_inv",
    "generate_mock",
    "has_noise_model",
    "uses_student_t",
    "variable_noise_hamiltonian",
]
```

- [ ] **Step 2: Move files with git mv (preserves history)**

```bash
git mv src/diffsed/model.py src/diffsed/core/model.py
git mv src/diffsed/param_spec.py src/diffsed/core/param_spec.py
git mv src/diffsed/_param_translate.py src/diffsed/core/param_translate.py
git mv src/diffsed/_fused_kernels.py src/diffsed/core/fused_kernels.py
git mv src/diffsed/_sed_pipeline.py src/diffsed/core/sed_pipeline.py
git mv src/diffsed/_mock.py src/diffsed/core/mock.py
git mv src/diffsed/prediction.py src/diffsed/core/prediction.py
git mv src/diffsed/noise.py src/diffsed/core/noise.py
```

- [ ] **Step 3: Update internal imports within core/ files**

In `core/model.py`, update:
```python
# Old:
from diffsed._fused_kernels import ...
from diffsed._param_translate import ...
from diffsed._sed_pipeline import ...
# New:
from diffsed.core.fused_kernels import ...
from diffsed.core.param_translate import ...
from diffsed.core.sed_pipeline import ...
```

Also update lazy imports inside methods:
- `from diffsed.prediction import ...` → `from diffsed.core.prediction import ...`
- `from diffsed.fitter import Fitter` → `from diffsed.inference.fitter import Fitter`

In `core/mock.py`, update:
```python
# Old: from diffsed.model import MockData
# New: from diffsed.core.model import MockData
```

In `core/param_spec.py` — imports `from diffsed.distributions` (stays at root, no change needed).

In `core/noise.py` — check for any internal imports that need updating.

In `core/prediction.py` — check for any internal imports.

In `core/fused_kernels.py` and `core/sed_pipeline.py` — check imports from `models/`, `utils/` (these don't change).

- [ ] **Step 4: Run tests (expect failures from import paths not yet updated elsewhere)**

```bash
source .venv/bin/activate && python -c "from diffsed.core import Model, ParamSpec"
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: create core/ subpackage, move model + param_spec + internals"
```

---

## Task 2: Move files into `inference/` subpackage

**Files:**
- Move: `fitter.py`, `hierarchical.py`, `posterior.py`, `raytrace_jax.py`, `vi_config.py`, `standardized.py` → `inference/`
- Update: `src/diffsed/inference/__init__.py`

- [ ] **Step 1: Move files with git mv**

```bash
git mv src/diffsed/fitter.py src/diffsed/inference/fitter.py
git mv src/diffsed/hierarchical.py src/diffsed/inference/hierarchical.py
git mv src/diffsed/posterior.py src/diffsed/inference/posterior.py
git mv src/diffsed/raytrace_jax.py src/diffsed/inference/raytrace.py
git mv src/diffsed/vi_config.py src/diffsed/inference/vi_config.py
git mv src/diffsed/standardized.py src/diffsed/inference/standardized.py
```

- [ ] **Step 2: Update `inference/__init__.py`**

```python
"""Inference engine: Fitter, Posterior, HierarchicalFitter, and backends."""

from diffsed.inference.fitter import Fitter
from diffsed.inference.hierarchical import HierarchicalFitter, HierarchicalResult
from diffsed.inference.posterior import Posterior
from diffsed.inference.raytrace import sample_raytrace
from diffsed.inference.vi_config import VIConfig

__all__ = [
    "Fitter",
    "HierarchicalFitter",
    "HierarchicalResult",
    "Posterior",
    "VIConfig",
    "sample_raytrace",
]
```

- [ ] **Step 3: Update internal imports within inference/ files**

In `inference/fitter.py`, update lazy imports:
```python
# Old:
from diffsed.noise import ...
from diffsed.posterior import Posterior
from diffsed.vi_config import VIConfig, evi_sample_mode
from diffsed.raytrace_jax import sample_raytrace
# New:
from diffsed.core.noise import ...
from diffsed.inference.posterior import Posterior
from diffsed.inference.vi_config import VIConfig, evi_sample_mode
from diffsed.inference.raytrace import sample_raytrace
```

In `inference/hierarchical.py`, update lazy imports:
```python
# Old:
from diffsed.vi_config import VIConfig, evi_sample_mode
from diffsed.raytrace_jax import sample_raytrace
# New:
from diffsed.inference.vi_config import VIConfig, evi_sample_mode
from diffsed.inference.raytrace import sample_raytrace
```

In `inference/posterior.py`, update lazy imports:
```python
# Old: from diffsed.param_spec import ParamSpec
# New: from diffsed.core.param_spec import ParamSpec
```

In `inference/standardized.py`, update:
```python
# Old: from diffsed.noise import ...
# New: from diffsed.core.noise import ...
```

In `inference/geovi.py`, update:
```python
# Old: from diffsed.noise import compute_std_inv
# New: from diffsed.core.noise import compute_std_inv
```

- [ ] **Step 4: Verify**

```bash
source .venv/bin/activate && python -c "from diffsed.inference import Fitter, Posterior"
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: move fitter, hierarchical, posterior, raytrace into inference/"
```

---

## Task 3: Update root `__init__.py`

**Files:**
- Modify: `src/diffsed/__init__.py`

- [ ] **Step 1: Update all imports to new paths**

The root `__init__.py` is the public API hub. Update every import to the new location:

```python
# Core
from diffsed.core.mock import MockData, generate_mock
from diffsed.core.model import Model
from diffsed.core.noise import (
    compute_effective_noise, compute_std_inv, has_noise_model,
    uses_student_t, variable_noise_hamiltonian,
)
from diffsed.core.param_spec import ParamSpec
from diffsed.core.param_translate import LOG10_ZSUN  # if exported
from diffsed.core.prediction import (
    DerivedQuantities, EmissionLines, Prediction, SEDQuantities, SFHQuantities,
)

# Inference
from diffsed.inference.fitter import Fitter
from diffsed.inference.hierarchical import HierarchicalFitter, HierarchicalResult
from diffsed.inference.posterior import Posterior
from diffsed.inference.raytrace import sample_raytrace
from diffsed.inference.vi_config import VIConfig

# These stay at root — no change:
from diffsed.distributions import Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform
from diffsed.plotting import ...
```

Keep `__all__` the same — the public API is unchanged.

- [ ] **Step 2: Verify public API works**

```bash
source .venv/bin/activate && python -c "
from diffsed import Model, ParamSpec, Fitter, Posterior, Uniform
from diffsed import Prediction, HierarchicalFitter, VIConfig
from diffsed import generate_mock, MockData, sample_raytrace
print('Public API OK')
"
```

- [ ] **Step 3: Also verify subpackage imports**

```bash
source .venv/bin/activate && python -c "
from diffsed.core import Model, ParamSpec
from diffsed.inference import Fitter, Posterior
print('Subpackage API OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: update __init__.py to import from core/ and inference/"
```

---

## Task 4: Update all test imports

**Files:**
- Modify: ~16 test files in `tests/`

- [ ] **Step 1: Bulk search-and-replace in tests/**

Apply these replacements across all test files:

```
from diffsed.model import          → from diffsed.core.model import
from diffsed.param_spec import     → from diffsed.core.param_spec import
from diffsed._param_translate import → from diffsed.core.param_translate import
from diffsed._mock import          → from diffsed.core.mock import
from diffsed.prediction import     → from diffsed.core.prediction import
from diffsed.noise import          → from diffsed.core.noise import
from diffsed.fitter import         → from diffsed.inference.fitter import
from diffsed.posterior import      → from diffsed.inference.posterior import
from diffsed.raytrace_jax import   → from diffsed.inference.raytrace import
from diffsed.vi_config import      → from diffsed.inference.vi_config import
from diffsed.hierarchical import   → from diffsed.inference.hierarchical import
from diffsed.standardized import   → from diffsed.inference.standardized import
```

Key test files to update:
- `tests/unit/test_param_translate.py`
- `tests/unit/test_model.py`
- `tests/unit/test_mock.py`
- `tests/unit/test_param_spec.py`
- `tests/unit/test_posterior.py`
- `tests/unit/test_fitter.py`
- `tests/unit/test_raytrace.py`
- `tests/unit/test_noise.py`
- `tests/unit/test_geovi_jit.py`
- `tests/unit/test_nebular_flags.py`
- `tests/unit/test_new_physics.py`
- `tests/unit/test_alpha_fe.py`
- `tests/unit/test_evolving_metallicity.py`
- `tests/crossval/test_physics_crossval.py`
- `tests/crossval/test_geovi_crossval.py`
- `tests/integration/test_model_integration.py`
- `tests/integration/test_derived_quantities.py`
- `tests/conftest.py` (if it imports any of these)

- [ ] **Step 2: Run full test suite**

```bash
source .venv/bin/activate && pytest tests/ -q
```

All ~808 tests must pass.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "refactor: update all test imports for core/ and inference/ paths"
```

---

## Task 5: Update notebook and script imports

**Files:**
- Modify: ~4 notebook files, any analysis/scripts that import moved modules

- [ ] **Step 1: Search and update notebooks**

```bash
grep -rl "from diffsed\.\(model\|param_spec\|_param_translate\|noise\|fitter\|posterior\|raytrace_jax\|vi_config\|standardized\|prediction\|_mock\|_fused_kernels\|_sed_pipeline\|hierarchical\)" notebooks/ --include="*.py"
```

Apply the same import mapping. Only edit `.py` files (not `.ipynb`).

Key files:
- `notebooks/archive/13_dust_models.py` — `from diffsed.model import LOG10_ZSUN`
- `notebooks/archive/11_noise_model.py` — `from diffsed.noise`
- `notebooks/archive/09_custom_models.py` — `from diffsed.standardized`
- `notebooks/reference/notebook_code/06_noise_models.py` — `from diffsed.noise`

- [ ] **Step 2: Check analysis/ and scripts/**

```bash
grep -rl "from diffsed\.\(model\|fitter\|param_spec\|noise\|posterior\)" analysis/ scripts/ --include="*.py"
```

Update any matches.

- [ ] **Step 3: Run ruff**

```bash
source .venv/bin/activate && ruff check src/ tests/ && ruff format src/ tests/
```

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor: update notebook and script imports for new package structure"
```

---

## Task 6: Update CLAUDE.md and AGENTS.md

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update package structure in CLAUDE.md**

Replace the package structure section with:

```
src/diffsed/
├── __init__.py              # public API re-exports
├── distributions.py         # Uniform, Gaussian, LogUniform, Fixed
├── plotting.py              # Visualization utilities
├── simulate.py              # SED-from-SFH utilities
│
├── core/                    # forward model
│   ├── model.py             # Model class (thin orchestrator)
│   ├── param_spec.py        # ParamSpec: parameter definitions + validation
│   ├── param_translate.py   # Public→internal param mapping + unit conversion
│   ├── fused_kernels.py     # JIT kernel factory functions
│   ├── sed_pipeline.py      # Core SED computation engine
│   ├── prediction.py        # Lazy Prediction object
│   ├── noise.py             # Noise model handling
│   └── mock.py              # Mock galaxy generation
│
├── inference/               # all fitting + results
│   ├── fitter.py            # Fitter: MAP, Ray Tracing, NUTS, geoVI, MGVI
│   ├── hierarchical.py      # HierarchicalFitter: shared PSD
│   ├── posterior.py          # Posterior: summary, corner, ESS
│   ├── raytrace.py          # Ray Tracing Sampler (Behroozi 2025)
│   ├── vi_config.py         # VI settings
│   ├── common.py            # Shared inference utilities
│   ├── nuts.py, geovi.py    # Backend implementations
│   └── map_optimizer.py     # MAP optimization
│
├── models/                  # physics modules
│   ├── sfh/                 # SFH models, PSD, GP generation
│   ├── dust/                # Two-component attenuation + IR emission
│   ├── agn/                 # AGN disc + torus models
│   ├── nebular/             # Nebular emission (BakedIn, CLOUDY, Cue)
│   ├── sps/                 # DSPS wrapper, SSP loading
│   ├── observation/         # Photometry, spectroscopy, filters
│   ├── igm.py, radio.py, xray.py
│
├── utils/                   # Grid, cosmology, transforms
└── diagnostics/             # Fisher, saliency, green functions
```

- [ ] **Step 2: Update AGENTS.md**

Update any import examples to use new paths.

- [ ] **Step 3: Commit**

```bash
git add -A && git commit -m "docs: update CLAUDE.md and AGENTS.md for new package structure"
```

---

## Verification

After all tasks:

```bash
# Full test suite
source .venv/bin/activate && pytest tests/ -q

# Lint
ruff check src/ tests/ && ruff format --check src/ tests/

# Public API (must work unchanged)
python -c "from diffsed import Model, ParamSpec, Fitter, Posterior, Uniform"

# Subpackage API (new convenience)
python -c "from diffsed.core import Model, ParamSpec"
python -c "from diffsed.inference import Fitter, Posterior"

# Verify old root files are gone
test ! -f src/diffsed/model.py && echo "model.py moved"
test ! -f src/diffsed/fitter.py && echo "fitter.py moved"
test ! -f src/diffsed/param_spec.py && echo "param_spec.py moved"

# Count root files (target: 4 .py files + __init__.py)
ls src/diffsed/*.py | wc -l
```
