# Package Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the flat tengri package into `core/` (forward model) and expanded `inference/` (fitting) subpackages, reducing root clutter from 18 to 3 files.

**Architecture:** Move files into subpackages, update all internal imports (strategy 1: clean break, no shims), keep public API unchanged via `__init__.py` re-exports. Subpackage `__init__.py` files also export key classes for convenience (`from tengri.core import Model`).

**Tech Stack:** Python, JAX, ruff, pytest

---

## Target Structure

```
src/tengri/
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
| `tengri.model` | `tengri.forward.sed_model` |
| `tengri.param_spec` | `tengri.core.param_spec` |
| `tengri._param_translate` | `tengri.parameters.translate` |
| `tengri._fused_kernels` | `tengri.forward.kernels.assembly` |
| `tengri._sed_pipeline` | `tengri.forward.pipeline` |
| `tengri._mock` | `tengri.observation.mock` |
| `tengri.prediction` | `tengri.forward.prediction` |
| `tengri.noise` | `tengri.observation.noise` |
| `tengri.fitter` | `tengri.inference.fitter` |
| `tengri.hierarchical` | `tengri.inference.hierarchical` |
| `tengri.posterior` | `tengri.inference.posterior` |
| `tengri.raytrace_jax` | `tengri.inference.raytrace` |
| `tengri.vi_config` | `tengri.inference.vi_config` |
| `tengri.standardized` | `tengri.inference.standardized` |

---

## Task 1: Create `core/` subpackage and move files

**Files:**
- Create: `src/tengri/core/__init__.py`
- Move: `model.py`, `param_spec.py`, `_param_translate.py`, `_fused_kernels.py`, `_sed_pipeline.py`, `_mock.py`, `prediction.py`, `noise.py` → `core/`
- Rename: drop `_` prefix on moved files

- [ ] **Step 1: Create core/ directory and __init__.py**

```bash
mkdir -p src/tengri/core
```

Create `src/tengri/core/__init__.py`:
```python
"""Core forward model: Model, ParamSpec, Prediction, and internals."""

from tengri.observation.mock import MockData, generate_mock
from tengri.forward.sed_model import Model
from tengri.observation.noise import (
    compute_effective_noise,
    compute_std_inv,
    has_noise_model,
    uses_student_t,
    variable_noise_hamiltonian,
)
from tengri.core.param_spec import ParamSpec
from tengri.parameters.translate import LOG10_ZSUN
from tengri.forward.prediction import (
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
git mv src/tengri/model.py src/tengri/core/model.py
git mv src/tengri/param_spec.py src/tengri/core/param_spec.py
git mv src/tengri/_param_translate.py src/tengri/core/param_translate.py
git mv src/tengri/_fused_kernels.py src/tengri/core/fused_kernels.py
git mv src/tengri/_sed_pipeline.py src/tengri/core/sed_pipeline.py
git mv src/tengri/_mock.py src/tengri/core/mock.py
git mv src/tengri/prediction.py src/tengri/core/prediction.py
git mv src/tengri/noise.py src/tengri/core/noise.py
```

- [ ] **Step 3: Update internal imports within core/ files**

In `core/model.py`, update:
```python
# Old:
from tengri._fused_kernels import ...
from tengri._param_translate import ...
from tengri._sed_pipeline import ...
# New:
from tengri.forward.kernels.assembly import ...
from tengri.parameters.translate import ...
from tengri.forward.pipeline import ...
```

Also update lazy imports inside methods:
- `from tengri.prediction import ...` → `from tengri.forward.prediction import ...`
- `from tengri.fitter import Fitter` → `from tengri.inference.fitter import Fitter`

In `core/mock.py`, update:
```python
# Old: from tengri.model import MockData
# New: from tengri.forward.sed_model import MockData
```

In `core/param_spec.py` — imports `from tengri.distributions` (stays at root, no change needed).

In `core/noise.py` — check for any internal imports that need updating.

In `core/prediction.py` — check for any internal imports.

In `core/fused_kernels.py` and `core/sed_pipeline.py` — check imports from `models/`, `utils/` (these don't change).

- [ ] **Step 4: Run tests (expect failures from import paths not yet updated elsewhere)**

```bash
source .venv/bin/activate && python -c "from tengri.core import Model, ParamSpec"
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: create core/ subpackage, move model + param_spec + internals"
```

---

## Task 2: Move files into `inference/` subpackage

**Files:**
- Move: `fitter.py`, `hierarchical.py`, `posterior.py`, `raytrace_jax.py`, `vi_config.py`, `standardized.py` → `inference/`
- Update: `src/tengri/inference/__init__.py`

- [ ] **Step 1: Move files with git mv**

```bash
git mv src/tengri/fitter.py src/tengri/inference/fitter.py
git mv src/tengri/hierarchical.py src/tengri/inference/hierarchical.py
git mv src/tengri/posterior.py src/tengri/inference/posterior.py
git mv src/tengri/raytrace_jax.py src/tengri/inference/raytrace.py
git mv src/tengri/vi_config.py src/tengri/inference/vi_config.py
git mv src/tengri/standardized.py src/tengri/inference/standardized.py
```

- [ ] **Step 2: Update `inference/__init__.py`**

```python
"""Inference engine: Fitter, Posterior, HierarchicalFitter, and backends."""

from tengri.inference.fitter import Fitter
from tengri.inference.hierarchical import HierarchicalFitter, HierarchicalResult
from tengri.inference.posterior import Posterior
from tengri.inference.raytrace import sample_raytrace
from tengri.inference.vi_config import VIConfig

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
from tengri.noise import ...
from tengri.posterior import Posterior
from tengri.vi_config import VIConfig, evi_sample_mode
from tengri.raytrace_jax import sample_raytrace
# New:
from tengri.observation.noise import ...
from tengri.inference.posterior import Posterior
from tengri.inference.vi_config import VIConfig, evi_sample_mode
from tengri.inference.raytrace import sample_raytrace
```

In `inference/hierarchical.py`, update lazy imports:
```python
# Old:
from tengri.vi_config import VIConfig, evi_sample_mode
from tengri.raytrace_jax import sample_raytrace
# New:
from tengri.inference.vi_config import VIConfig, evi_sample_mode
from tengri.inference.raytrace import sample_raytrace
```

In `inference/posterior.py`, update lazy imports:
```python
# Old: from tengri.param_spec import ParamSpec
# New: from tengri.core.param_spec import ParamSpec
```

In `inference/standardized.py`, update:
```python
# Old: from tengri.noise import ...
# New: from tengri.observation.noise import ...
```

In `inference/geovi.py`, update:
```python
# Old: from tengri.noise import compute_std_inv
# New: from tengri.observation.noise import compute_std_inv
```

- [ ] **Step 4: Verify**

```bash
source .venv/bin/activate && python -c "from tengri.inference import Fitter, Posterior"
```

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "refactor: move fitter, hierarchical, posterior, raytrace into inference/"
```

---

## Task 3: Update root `__init__.py`

**Files:**
- Modify: `src/tengri/__init__.py`

- [ ] **Step 1: Update all imports to new paths**

The root `__init__.py` is the public API hub. Update every import to the new location:

```python
# Core
from tengri.observation.mock import MockData, generate_mock
from tengri.forward.sed_model import Model
from tengri.observation.noise import (
    compute_effective_noise, compute_std_inv, has_noise_model,
    uses_student_t, variable_noise_hamiltonian,
)
from tengri.core.param_spec import ParamSpec
from tengri.parameters.translate import LOG10_ZSUN  # if exported
from tengri.forward.prediction import (
    DerivedQuantities, EmissionLines, Prediction, SEDQuantities, SFHQuantities,
)

# Inference
from tengri.inference.fitter import Fitter
from tengri.inference.hierarchical import HierarchicalFitter, HierarchicalResult
from tengri.inference.posterior import Posterior
from tengri.inference.raytrace import sample_raytrace
from tengri.inference.vi_config import VIConfig

# These stay at root — no change:
from tengri.distributions import Fixed, Gaussian, LogNormal, LogUniform, StudentT, Uniform
from tengri.plotting import ...
```

Keep `__all__` the same — the public API is unchanged.

- [ ] **Step 2: Verify public API works**

```bash
source .venv/bin/activate && python -c "
from tengri import Model, ParamSpec, Fitter, Posterior, Uniform
from tengri import Prediction, HierarchicalFitter, VIConfig
from tengri import generate_mock, MockData, sample_raytrace
print('Public API OK')
"
```

- [ ] **Step 3: Also verify subpackage imports**

```bash
source .venv/bin/activate && python -c "
from tengri.core import Model, ParamSpec
from tengri.inference import Fitter, Posterior
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
from tengri.model import          → from tengri.forward.sed_model import
from tengri.param_spec import     → from tengri.core.param_spec import
from tengri._param_translate import → from tengri.parameters.translate import
from tengri._mock import          → from tengri.observation.mock import
from tengri.prediction import     → from tengri.forward.prediction import
from tengri.noise import          → from tengri.observation.noise import
from tengri.fitter import         → from tengri.inference.fitter import
from tengri.posterior import      → from tengri.inference.posterior import
from tengri.raytrace_jax import   → from tengri.inference.raytrace import
from tengri.vi_config import      → from tengri.inference.vi_config import
from tengri.hierarchical import   → from tengri.inference.hierarchical import
from tengri.standardized import   → from tengri.inference.standardized import
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
grep -rl "from tengri\.\(model\|param_spec\|_param_translate\|noise\|fitter\|posterior\|raytrace_jax\|vi_config\|standardized\|prediction\|_mock\|_fused_kernels\|_sed_pipeline\|hierarchical\)" notebooks/ --include="*.py"
```

Apply the same import mapping. Only edit `.py` files (not `.ipynb`).

Key files:
- `notebooks/archive/13_dust_models.py` — `from tengri.model import LOG10_ZSUN`
- `notebooks/archive/11_noise_model.py` — `from tengri.noise`
- `notebooks/archive/09_custom_models.py` — `from tengri.standardized`
- `notebooks/reference/notebook_code/06_noise_models.py` — `from tengri.noise`

- [ ] **Step 2: Check analysis/ and scripts/**

```bash
grep -rl "from tengri\.\(model\|fitter\|param_spec\|noise\|posterior\)" analysis/ scripts/ --include="*.py"
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
src/tengri/
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
python -c "from tengri import Model, ParamSpec, Fitter, Posterior, Uniform"

# Subpackage API (new convenience)
python -c "from tengri.core import Model, ParamSpec"
python -c "from tengri.inference import Fitter, Posterior"

# Verify old root files are gone
test ! -f src/tengri/model.py && echo "model.py moved"
test ! -f src/tengri/fitter.py && echo "fitter.py moved"
test ! -f src/tengri/param_spec.py && echo "param_spec.py moved"

# Count root files (target: 4 .py files + __init__.py)
ls src/tengri/*.py | wc -l
```
