# Nebular Backend Flags Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the implicit `nebular` flag with three explicit backend flags (`nebular_ssp`, `nebular`, `nebular_cue`) and add ionization source control for Cue.

**Architecture:** ParamSpec validates flags and resolves the backend type into `self.nebular_mode` (one of `"off"`, `"ssp"`, `"cloudy"`, `"cue"`). Model reads `nebular_mode` to dispatch the correct backend. Cue gets a default weights path and optional ionspec param registration.

**Tech Stack:** Python, JAX, pytest

**Spec:** `docs/superpowers/specs/2026-03-20-nebular-backend-flags-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/tengri/param_spec.py` | Modify | Flag parsing, validation, param registration |
| `src/tengri/model.py` | Modify | Backend dispatch using `nebular_mode` |
| `src/tengri/models/nebular/__init__.py` | Modify | Default Cue weights path constant |
| `tests/unit/test_nebular_flags.py` | Create | All flag validation and param registration tests |

---

### Task 1: Default Cue Weights Path

**Files:**
- Modify: `src/tengri/models/nebular/__init__.py`
- Modify: `src/tengri/models/nebular/cue.py`

- [ ] **Step 1: Add default path constant to `__init__.py`**

In `src/tengri/models/nebular/__init__.py`, add after imports:

```python
from pathlib import Path

_DEFAULT_CUE_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "cue_weights.npz"
)
```

- [ ] **Step 2: Verify the path resolves correctly**

Run: `python -c "from tengri.models.nebular import _DEFAULT_CUE_WEIGHTS_PATH; print(_DEFAULT_CUE_WEIGHTS_PATH); print(_DEFAULT_CUE_WEIGHTS_PATH.exists())"`

Expected: path to `data/cue_weights.npz`, `True`

- [ ] **Step 3: Commit**

```bash
git add src/tengri/models/nebular/__init__.py
git commit -m "chore: add default Cue weights path constant"
```

---

### Task 2: Add Cue Ionspec Param Definitions

**Files:**
- Modify: `src/tengri/param_spec.py`

- [ ] **Step 1: Add `_CUE_IONSPEC_PARAMS` and `_CUE_GAS_EXTRA_PARAMS` dicts**

After `_NEBULAR_PARAMS` (~line 136), add:

```python
# Cue-specific optional params — only registered if user provides them
_CUE_IONSPEC_PARAMS = {
    "ionspec_index1": (
        "Cue ionizing spectrum slope segment 1 (HeII, 1-228A)",
        lambda lo, hi: 0 <= lo and hi <= 50,
        "must be in [0, 50]",
        None,  # None = not registered by default
    ),
    "ionspec_index2": (
        "Cue ionizing spectrum slope segment 2 (OII, 228-353A)",
        lambda lo, hi: -1 <= lo and hi <= 35,
        "must be in [-1, 35]",
        None,
    ),
    "ionspec_index3": (
        "Cue ionizing spectrum slope segment 3 (HeI, 353-504A)",
        lambda lo, hi: -2 <= lo and hi <= 20,
        "must be in [-2, 20]",
        None,
    ),
    "ionspec_index4": (
        "Cue ionizing spectrum slope segment 4 (HI, 504-912A)",
        lambda lo, hi: -2 <= lo and hi <= 10,
        "must be in [-2, 10]",
        None,
    ),
    "ionspec_logLratio1": (
        "Cue log luminosity ratio seg2/seg1",
        lambda lo, hi: -1 <= lo and hi <= 12,
        "must be in [-1, 12]",
        None,
    ),
    "ionspec_logLratio2": (
        "Cue log luminosity ratio seg3/seg2",
        lambda lo, hi: -1 <= lo and hi <= 3,
        "must be in [-1, 3]",
        None,
    ),
    "ionspec_logLratio3": (
        "Cue log luminosity ratio seg4/seg3",
        lambda lo, hi: -1 <= lo and hi <= 3,
        "must be in [-1, 3]",
        None,
    ),
}

_CUE_GAS_EXTRA_PARAMS = {
    "gas_logn": (
        "Cue gas density log10(n_H/cm^-3)",
        lambda lo, hi: 0 <= lo and hi <= 5,
        "must be in [0, 5]",
        None,
    ),
    "gas_logno": (
        "Cue [N/O] abundance ratio (dex)",
        lambda lo, hi: -2 <= lo and hi <= 2,
        "must be in [-2, 2]",
        None,
    ),
    "gas_logco": (
        "Cue [C/O] abundance ratio (dex)",
        lambda lo, hi: -2 <= lo and hi <= 2,
        "must be in [-2, 2]",
        None,
    ),
}
```

- [ ] **Step 2: Add these to `SETTINGS_KEYS`**

Add `"nebular_ssp"`, `"nebular_cue"`, `"neb_ionization"` to `SETTINGS_KEYS` frozenset. Keep `"nebular"`, `"cloudy_grid_path"`, `"cue_weights_path"`.

- [ ] **Step 3: Commit**

```bash
git add src/tengri/param_spec.py
git commit -m "feat: add Cue ionspec + gas param definitions to param_spec"
```

---

### Task 3: Write Tests for Flag Validation

**Files:**
- Create: `tests/unit/test_nebular_flags.py`

- [ ] **Step 1: Write all flag validation tests**

```python
"""Tests for nebular backend flag validation in ParamSpec."""

import warnings
from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.distributions import Fixed, Uniform
from tengri.param_spec import ParamSpec


class TestNebularFlagConflicts:
    """Mutually exclusive flag validation."""

    def test_no_nebular_is_default(self):
        spec = ParamSpec()
        assert spec.nebular_mode == "off"

    def test_nebular_ssp_sets_mode(self):
        spec = ParamSpec(nebular_ssp=True)
        assert spec.nebular_mode == "ssp"

    def test_nebular_cloudy_requires_grid_path(self):
        with pytest.raises(ValueError, match="cloudy_grid_path"):
            ParamSpec(nebular=True)

    def test_nebular_cloudy_with_path(self):
        spec = ParamSpec(
            nebular=True,
            cloudy_grid_path="data/cloudy_grid_mist.h5",
        )
        assert spec.nebular_mode == "cloudy"

    def test_nebular_cue_sets_mode(self):
        spec = ParamSpec(nebular_cue=True)
        assert spec.nebular_mode == "cue"

    def test_nebular_cue_default_weights_path(self):
        spec = ParamSpec(nebular_cue=True)
        assert spec.cue_weights_path is not None
        assert "cue_weights" in str(spec.cue_weights_path)

    def test_nebular_cue_custom_weights_path(self):
        spec = ParamSpec(nebular_cue=True, cue_weights_path="/custom/path.npz")
        assert spec.cue_weights_path == "/custom/path.npz"

    def test_conflict_ssp_and_cloudy(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ParamSpec(nebular_ssp=True, nebular=True,
                      cloudy_grid_path="x.h5")

    def test_conflict_ssp_and_cue(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ParamSpec(nebular_ssp=True, nebular_cue=True)

    def test_conflict_cloudy_and_cue(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            ParamSpec(nebular=True, nebular_cue=True,
                      cloudy_grid_path="x.h5")


class TestNebularSspWarnings:
    """BakedIn mode warns when user sets nebular params."""

    def test_warns_on_neb_logU(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ParamSpec(nebular_ssp=True, neb_logU=Uniform(-4, -1))
            neb_warnings = [x for x in w if "nebular_ssp" in str(x.message)]
            assert len(neb_warnings) > 0

    def test_no_neb_params_registered(self):
        spec = ParamSpec(nebular_ssp=True)
        assert "neb_logU" not in spec.free_params
        assert "neb_logU" not in spec.all_params


class TestNebularCloudyParams:
    """CLOUDY mode registers standard nebular params."""

    def test_registers_neb_params(self):
        spec = ParamSpec(nebular=True,
                         cloudy_grid_path="data/cloudy_grid_mist.h5")
        assert "neb_logU" in spec.all_params
        assert "neb_fesc" in spec.all_params

    def test_does_not_register_ionspec(self):
        spec = ParamSpec(nebular=True,
                         cloudy_grid_path="data/cloudy_grid_mist.h5")
        assert "ionspec_index1" not in spec.all_params


class TestNebularCueParams:
    """Cue mode registers nebular params + optional ionspec."""

    def test_registers_standard_neb_params(self):
        spec = ParamSpec(nebular_cue=True)
        assert "neb_logU" in spec.all_params
        assert "neb_fesc" in spec.all_params

    def test_ionspec_not_registered_by_default(self):
        spec = ParamSpec(nebular_cue=True)
        assert "ionspec_index1" not in spec.all_params
        assert "ionspec_index1" not in spec.free_params

    def test_ionspec_fixed_registered(self):
        spec = ParamSpec(nebular_cue=True, ionspec_index1=Fixed(5.0))
        assert "ionspec_index1" in spec.all_params
        assert "ionspec_index1" not in spec.free_params

    def test_ionspec_free_registered(self):
        spec = ParamSpec(nebular_cue=True,
                         ionspec_index1=Uniform(1, 42))
        assert "ionspec_index1" in spec.free_params

    def test_gas_extra_params_optional(self):
        spec = ParamSpec(nebular_cue=True, gas_logn=Fixed(2.5))
        assert "gas_logn" in spec.all_params

    def test_ionspec_on_cloudy_raises(self):
        """ionspec params on non-Cue backend should raise."""
        with pytest.raises(ValueError, match="ionspec"):
            ParamSpec(nebular=True, cloudy_grid_path="x.h5",
                      ionspec_index1=Uniform(1, 42))


class TestNebularIonization:
    """neb_ionization flag for Cue."""

    def test_default_is_ssp(self):
        spec = ParamSpec(nebular_cue=True)
        assert spec.neb_ionization == "ssp"

    def test_agn_not_implemented(self):
        with pytest.raises(NotImplementedError, match="AGN ionization"):
            ParamSpec(nebular_cue=True, neb_ionization="agn")

    def test_ssp_agn_not_implemented(self):
        with pytest.raises(NotImplementedError, match="AGN ionization"):
            ParamSpec(nebular_cue=True, neb_ionization="ssp+agn")


class TestBackwardCompat:
    """Old-style nebular flags still work."""

    def test_nebular_string_cue(self):
        spec = ParamSpec(nebular="cue")
        assert spec.nebular_mode == "cue"

    def test_cue_weights_path_implies_cue(self):
        spec = ParamSpec(cue_weights_path="data/cue_weights.npz")
        assert spec.nebular_mode == "cue"

    def test_cloudy_grid_path_implies_cloudy(self):
        spec = ParamSpec(cloudy_grid_path="data/cloudy_grid_mist.h5")
        assert spec.nebular_mode == "cloudy"
```

- [ ] **Step 2: Run tests — should all fail (RED)**

Run: `JAX_PLATFORMS=cpu pytest tests/unit/test_nebular_flags.py -v --tb=short`

Expected: all fail (no `nebular_mode`, no `nebular_ssp`, etc.)

- [ ] **Step 3: Commit test file**

```bash
git add tests/unit/test_nebular_flags.py
git commit -m "test: nebular flag validation tests (RED)"
```

---

### Task 4: Implement Flag Parsing in ParamSpec

**Files:**
- Modify: `src/tengri/param_spec.py:389-414` (SETTINGS_KEYS)
- Modify: `src/tengri/param_spec.py:423-484` (_build_param_registry)
- Modify: `src/tengri/param_spec.py:748-756` (ParamSpec.__init__)

- [ ] **Step 1: Update SETTINGS_KEYS**

Add `"nebular_ssp"`, `"nebular_cue"`, `"neb_ionization"` to the frozenset. Keep existing keys.

- [ ] **Step 2: Rewrite nebular section in ParamSpec.__init__**

Replace lines 748-756 with:

```python
        # --- Nebular emission ---
        nebular_ssp = kwargs.pop("nebular_ssp", False)
        nebular = kwargs.pop("nebular", False)
        nebular_cue = kwargs.pop("nebular_cue", False)
        self.cloudy_grid_path = kwargs.pop("cloudy_grid_path", None)
        self.cue_weights_path = kwargs.pop("cue_weights_path", None)
        self.neb_ionization = kwargs.pop("neb_ionization", "ssp")

        # Backward compat: old string-style flags
        if nebular == "cue":
            nebular_cue = True
            nebular = False
        elif nebular == "cloudy":
            nebular = True

        # Backward compat: path implies backend
        if self.cue_weights_path is not None and not nebular_cue and not nebular:
            nebular_cue = True
        if self.cloudy_grid_path is not None and not nebular and not nebular_cue:
            nebular = True

        # Mutual exclusion check
        n_set = sum([bool(nebular_ssp), bool(nebular), bool(nebular_cue)])
        if n_set > 1:
            raise ValueError(
                "nebular_ssp, nebular (CLOUDY), and nebular_cue are "
                "mutually exclusive — choose one."
            )

        # Resolve mode
        if nebular_cue:
            self.nebular_mode = "cue"
            if self.cue_weights_path is None:
                from tengri.models.nebular import _DEFAULT_CUE_WEIGHTS_PATH
                self.cue_weights_path = str(_DEFAULT_CUE_WEIGHTS_PATH)
        elif nebular:
            self.nebular_mode = "cloudy"
            if self.cloudy_grid_path is None:
                self._raise_missing_grid_path()
        elif nebular_ssp:
            self.nebular_mode = "ssp"
        else:
            self.nebular_mode = "off"

        # Keep self.nebular as truthy for backward compat with Model
        self.nebular = self.nebular_mode != "off"

        # Validate ionization source
        if self.neb_ionization in ("agn", "ssp+agn"):
            raise NotImplementedError(
                "AGN ionization not yet implemented — use neb_ionization='ssp'"
            )

        # Warn if nebular_ssp user sets nebular params
        if self.nebular_mode == "ssp":
            _NEB_PARAM_NAMES = set(_NEBULAR_PARAMS) | set(_CUE_IONSPEC_PARAMS) | set(_CUE_GAS_EXTRA_PARAMS)
            for name in list(kwargs):
                if name in _NEB_PARAM_NAMES:
                    import warnings
                    warnings.warn(
                        f"'{name}' is ignored with nebular_ssp=True "
                        f"(emission is baked into SSP at fixed logU/logZ).",
                        UserWarning,
                        stacklevel=2,
                    )
                    kwargs.pop(name)
```

- [ ] **Step 3: Add `_raise_missing_grid_path` method**

```python
    @staticmethod
    def _raise_missing_grid_path():
        """Raise ValueError listing available CLOUDY grids."""
        from pathlib import Path
        data_dir = Path(__file__).resolve().parents[1] / "data"
        grids = sorted(data_dir.glob("cloudy_grid_*.h5"))
        grid_list = "\n".join(f"  {g.name}" for g in grids) if grids else "  (none found)"
        raise ValueError(
            f"nebular=True requires cloudy_grid_path. "
            f"Available grids in {data_dir}/:\n{grid_list}\n"
            f"Match the grid isochrone to your SSP for consistency."
        )
```

- [ ] **Step 4: Update `_build_param_registry` nebular section**

Replace the nebular block (~line 480-484) with:

```python
    # Nebular params (CLOUDY or Cue — not BakedIn)
    if nebular and nebular != "ssp":
        for pname, (desc, check, err, default) in _NEBULAR_PARAMS.items():
            registry[pname] = (desc, check, err)
            defaults[pname] = default
```

And pass `nebular` through as a string (`nebular_mode`) from `__init__`.

- [ ] **Step 5: Handle Cue-specific optional params in __init__**

After the nebular param registration, detect and register ionspec/gas params that the user explicitly provided:

```python
        # Cue-optional params: only register if user explicitly provides them
        if self.nebular_mode == "cue":
            for pname, (desc, check, err, _) in _CUE_IONSPEC_PARAMS.items():
                if pname in resolved_kwargs:
                    # User provided this param — register it
                    pass  # handled by normal param processing below
            for pname, (desc, check, err, _) in _CUE_GAS_EXTRA_PARAMS.items():
                if pname in resolved_kwargs:
                    pass
        elif self.nebular_mode in ("cloudy", "ssp", "off"):
            # Ionspec params on non-Cue backend → error
            for pname in _CUE_IONSPEC_PARAMS:
                if pname in resolved_kwargs:
                    raise ValueError(
                        f"'{pname}' is a Cue-only parameter. "
                        f"Use nebular_cue=True to enable the Cue emulator."
                    )
```

- [ ] **Step 6: Run tests — should pass (GREEN)**

Run: `JAX_PLATFORMS=cpu pytest tests/unit/test_nebular_flags.py -v --tb=short`

Expected: all pass

- [ ] **Step 7: Run full unit tests for regressions**

Run: `JAX_PLATFORMS=cpu pytest tests/unit/ -q --tb=short`

Expected: all pass (591+)

- [ ] **Step 8: Commit**

```bash
git add src/tengri/param_spec.py
git commit -m "feat: nebular backend flags (nebular_ssp, nebular, nebular_cue)"
```

---

### Task 5: Update Model Backend Dispatch

**Files:**
- Modify: `src/tengri/model.py:352-375`

- [ ] **Step 1: Simplify Model.__init__ nebular dispatch**

Replace the current nebular backend block with:

```python
        # Nebular emission backend + params
        if spec.nebular_mode in ("cloudy", "cue"):
            self._param_map["neb_logU"] = ("neb_logU", 1.0, 0.0)
            self._param_map["neb_logZ_gas"] = ("neb_logZ_gas", 1.0, LOG10_ZSUN)
            self._param_map["neb_fesc"] = ("neb_fesc", 1.0, 0.0)
            self._param_map["neb_fesc_lya"] = ("neb_fesc_lya", 1.0, 0.0)

        self._nebular_backend = None
        if spec.nebular_mode == "cue":
            from tengri.models.nebular import CueBackend
            self._nebular_backend = CueBackend(
                spec.cue_weights_path, ssp_data=ssp_data
            )
        elif spec.nebular_mode == "cloudy":
            from tengri.models.nebular import CloudyGridBackend
            self._nebular_backend = CloudyGridBackend(
                spec.cloudy_grid_path, ssp_data
            )
        elif spec.nebular_mode == "ssp":
            from tengri.models.nebular import BakedInBackend
            self._nebular_backend = BakedInBackend()
        else:
            from tengri.models.nebular import BakedInBackend
            self._nebular_backend = BakedInBackend()
```

- [ ] **Step 2: Run full test suite**

Run: `JAX_PLATFORMS=cpu pytest tests/unit/ -q --tb=short`

Expected: all pass

- [ ] **Step 3: Run crossval tests**

Run: `JAX_PLATFORMS=cpu pytest tests/crossval/test_nebular_crossval.py -m crossval -q`

Expected: 20 passed

- [ ] **Step 4: Commit**

```bash
git add src/tengri/model.py
git commit -m "refactor: Model nebular dispatch uses nebular_mode"
```

---

### Task 6: Update Docstrings and Examples

**Files:**
- Modify: `src/tengri/param_spec.py` (docstrings)

- [ ] **Step 1: Update ParamSpec class docstring**

Replace the "Nebular Emission Settings" section with:

```
    Nebular Emission Settings
    ~~~~~~~~~~~~~~~~~~~~~~~~~
    nebular_ssp : bool
        Use SSP files with pre-included nebular emission (wNE files).
        No free nebular parameters.  Default: ``False``.
    nebular : bool
        Enable CLOUDY grid nebular emission.  Requires ``cloudy_grid_path``.
        Default: ``False``.
    nebular_cue : bool
        Enable Cue neural emulator.  Default weights loaded automatically.
        Default: ``False``.
    cloudy_grid_path : str
        Path to CLOUDY HDF5 grid.  Required when ``nebular=True``.
    cue_weights_path : str
        Override default Cue weights path.
    neb_ionization : str
        Ionization source for Cue: ``"ssp"`` (default), ``"agn"`` (future),
        ``"ssp+agn"`` (future).
```

- [ ] **Step 2: Update the "Full model" example in docstring**

Replace `cue_weights_path="data/cue_weights.npz"` with `nebular_cue=True`.

- [ ] **Step 3: Run ruff check and format**

```bash
ruff check src/tengri/param_spec.py src/tengri/model.py
ruff format src/tengri/param_spec.py src/tengri/model.py
```

- [ ] **Step 4: Final full test run**

```bash
JAX_PLATFORMS=cpu pytest tests/unit/ -q --tb=short
JAX_PLATFORMS=cpu pytest tests/crossval/test_nebular_crossval.py tests/crossval/test_cue_crossval.py -m crossval -q
```

- [ ] **Step 5: Commit**

```bash
git add src/tengri/param_spec.py src/tengri/model.py
git commit -m "docs: update ParamSpec docstrings for nebular flags"
```
