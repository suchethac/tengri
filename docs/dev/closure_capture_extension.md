# Phase II-6: Registry-Driven Component Extension

## Overview

Phase II-6 introduces a **component registry** that enables adding new SED components (dust IR backends, AGN torus models, etc.) without modifying:

- `src/tengri/forward/sed_model.py` (orchestrator)
- `src/tengri/forward/_kernels/hybrid.py` (JIT kernel)
- `src/tengri/forward/sed_model_types.py` (PrecomputedData)

Adding a new component now requires:

1. Write a precompute function
2. Write a JIT-compiled lookup callable
3. Register the pair in one place

## Motivation

Before Phase II-6, adding a new dust IR backend or AGN torus model required edits in multiple places:

```python
# OLD PATTERN (Phase II-5)

# 1. Add field to PrecomputedData
# src/tengri/forward/sed_model_types.py
@dataclasses.dataclass(frozen=True)
class PrecomputedData:
    my_component_grid_arrays: tuple | None = None

# 2. Wire precompute in SEDModel.__init__
# src/tengri/forward/sed_model.py
my_preint, my_grid_arrays = self._precompute_my_component()

# 3. Add if/elif to hybrid kernel
# src/tengri/forward/_kernels/hybrid.py
if _has_preint_my_component:
    if _my_component_model_name == "backend1":
        my_phot = _my_lookup(param1, param2, grid_arrays_traced=_my_grid_arrays)
    elif _my_component_model_name == "backend2":
        my_phot = _my_lookup_backend2(param1, grid_arrays_traced=_my_grid_arrays)
    # ... more backends ...
```

This pattern is error-prone and doesn't scale.

## New Pattern (Phase II-6)

```python
# NEW PATTERN (Phase II-6)

# 1. Write precompute and lookup in component module
# src/tengri/components/my_component/my_precompute.py

def precompute_my_thing(...):
    """Precompute grids and return dict with grid_phot, axes."""
    return {"grid_phot": grid, "axes": axes}

def build_my_lookup(precomp, *, grid_arrays_traced=None):
    """Build JIT-compiled lookup callable."""
    @jax.jit
    def lookup(param1, param2, grid_arrays_traced=None):
        # triweight interpolation, etc.
        return phot
    return lookup

# 2. Register once at module load
_register_my_component()  # Automatically called

# That's it! No other files need editing.
```

## Component Registry API

### ComponentSpec

Each component is a frozen `ComponentSpec` with:

```python
@dataclass(frozen=True)
class ComponentSpec:
    name: str  # e.g., "dust_ir:my_backend"
    precompute: Callable  # (filters, trans, z, params) -> dict
    extract_arrays: Callable  # (precomp_dict) -> tuple of arrays
    build_lookup: Callable  # (precomp_dict, *, grid_arrays_traced) -> JIT callable
    apply_signature: tuple[str, ...]  # parameter names in positional order
    activation: Callable[[spec, parameters], bool]  # does this component fire?
```

### Registration

```python
from tengri.forward._component_registry import register, ComponentSpec

spec = ComponentSpec(
    name="dust_ir:my_backend",
    precompute=precompute_my_backend,
    extract_arrays=extract_my_arrays,
    build_lookup=build_my_lookup,
    apply_signature=("param1", "param2"),
    activation=lambda spec, params: params._dust_emission_model == "my_backend",
)
register(spec)
```

### Lookup / List

```python
from tengri.forward._component_registry import get_component, list_components

# Retrieve a single component
spec = get_component("dust_ir:draine_li2007")

# List all dust IR components
dust_comps = list_components(family="dust_ir")

# List all components
all_comps = list_components()
```

## Grid Arrays and JIT Threading

The registry design separates **precomputed grids** (potentially large) from **lightweight parameters**. Instead of closure-capturing 100 MB arrays in a JIT-compiled function, we thread them as **JIT-traced runtime arguments**.

### Extract Arrays

When precomputing, `extract_arrays` returns a tuple of JAX arrays suitable for JIT threading:

```python
def extract_my_arrays(precomp):
    """Return arrays that will be passed to lookup as grid_arrays_traced."""
    return (
        jnp.asarray(precomp["grid_phot"]),
        jnp.asarray(precomp["axes"]),
    )
```

### Build Lookup

The lookup function accepts `grid_arrays_traced` as an optional kwarg:

```python
def build_my_lookup(precomp, *, grid_arrays_traced=None):
    @jax.jit
    def lookup(param1, param2, grid_arrays_traced=None):
        if grid_arrays_traced is not None:
            grid, axes = grid_arrays_traced
        else:
            # Fallback for backward compatibility
            grid = precomp["grid_phot"]
            axes = precomp["axes"]
        return triweight_interp(grid, axes, (param1, param2))
    return lookup
```

At runtime, the hybrid kernel calls:

```python
# In _kernels/hybrid.py:
phot = lookup(param1, param2, grid_arrays_traced=traced_arrays)
```

The JIT compiler sees `traced_arrays` as a **runtime input** (not a closure constant), avoiding the large compile-time constant.

## Applied Components (Phase II-6)

Six dust IR models and SKIRTOR are registered:

**Dust IR:**
- `dust_ir:draine_li2007` — 2D (qpah, umin)
- `dust_ir:draine_li2014` — 3D (qpah, umin, alpha_dl14)
- `dust_ir:dale2014` — 1D (alpha)
- `dust_ir:astrodust` — 2D (qpah, umin)
- `dust_ir:themis` — 2D (qhac, umin)
- `dust_ir:bosa` — 1D (log_ssfr)

**AGN Torus:**
- `agn:skirtor` — 5D (tau, p, q, oa, cos_inc)

## Component Grid Arrays Storage

At `SEDModel.__init__`, precomputed arrays are stored in a generic dict:

```python
# In PrecomputedData:
component_grid_arrays: dict[str, tuple] = dataclasses.field(default_factory=dict)
```

For backward compatibility, per-component fields still exist:
- `dust_ir_grid_arrays` → `component_grid_arrays.get("dust_ir")`
- `skirtor_grid_arrays` → `component_grid_arrays.get("agn:skirtor")`

## Adding a New Component

Here's a worked example: adding a fictional **"my_dust"** dust emission backend.

### Step 1: Write precompute and lookup

File: `src/tengri/components/dust/my_dust_precompute.py`

```python
from tengri.utils.grid_interp import preintegrate_grid, interp_nd_triweight, edges_for_grid
import jax
import jax.numpy as jnp
import numpy as np

def precompute_my_dust(...):
    """Load templates, preintegrate through filters, return dict."""
    # Load your templates from disk
    templates = load_my_dust_templates()
    
    # Preintegrate through filters
    preint = preintegrate_grid(
        templates=np.asarray(templates["L_nu"]),
        wave_rest=np.asarray(templates["wavelength"]),
        filter_waves=[np.asarray(fw) for fw in filter_waves],
        filter_trans=[np.asarray(ft) for ft in filter_trans],
        redshift=redshift,
        dl_cm=1.0,
        axes=(np.asarray(templates["param1_axis"]),),
        energy_normalize=True,
    )
    
    return {
        "grid_phot": preint.phot,
        "axes": (jnp.asarray(templates["param1_axis"]),),
    }

def extract_my_dust_arrays(precomp):
    """Return arrays for JIT threading."""
    return (
        jnp.asarray(precomp["grid_phot"]),
        precomp["axes"],
    )

def build_my_dust_lookup(precomp, *, grid_arrays_traced=None):
    """Build JIT-compiled photometry lookup."""
    if grid_arrays_traced is None:
        grid = precomp["grid_phot"]
        axes = precomp["axes"]
    else:
        grid, axes = grid_arrays_traced
    
    @jax.jit
    def my_dust_phot(L_absorbed, param1, grid_arrays_traced=None):
        if grid_arrays_traced is not None:
            grid, axes = grid_arrays_traced
        edges = tuple(edges_for_grid(ax) for ax in axes)
        phot_per_lsun = interp_nd_triweight(grid, axes, edges, (param1,))
        return L_absorbed * phot_per_lsun
    
    return my_dust_phot
```

### Step 2: Register

At the end of the same file:

```python
def _register_my_dust():
    from tengri.forward._component_registry import ComponentSpec, register
    
    def _precompute(filter_waves, filter_trans, redshift, parameters):
        try:
            return precompute_my_dust(filter_waves, filter_trans, redshift)
        except:
            return None
    
    def _activate(spec, parameters):
        return (
            hasattr(parameters, "_dust_emission_model")
            and parameters._dust_emission_model == "my_dust"
        )
    
    register(
        ComponentSpec(
            name="dust_ir:my_dust",
            precompute=_precompute,
            extract_arrays=extract_my_dust_arrays,
            build_lookup=build_my_dust_lookup,
            apply_signature=("param1",),
            activation=_activate,
        )
    )

_register_my_dust()
```

### Step 3: Done

No other files need editing. The component is automatically available:

```python
from tengri import SEDModel, Parameters

spec = Parameters(
    dust_emission_model="my_dust",
    my_dust_param1=Uniform(0.1, 1.0),
    ...
)
model = SEDModel(spec, ssp_data, filters)
# Precompute runs automatically, grid_arrays are stored in
# model._state.precomputed.component_grid_arrays["dust_ir"]
```

## Testing

Use `tests/unit/test_component_registry.py` as a template. Key checks:

1. Registry entries exist at module load
2. Activation predicates work correctly
3. Apply signatures match parameter order
4. Grid arrays can be extracted

Example:

```python
from tengri.forward._component_registry import get_component

def test_my_dust_registered():
    spec = get_component("dust_ir:my_dust")
    assert spec is not None
    assert spec.apply_signature == ("param1",)

def test_my_dust_activation():
    class MockParams:
        _dust_emission_model = "my_dust"
    
    spec = get_component("dust_ir:my_dust")
    assert spec.activation(spec, MockParams()) is True
```

## Future Work

- **Nested components**: Some components (e.g., AGN composite) may need to call other components. Use `get_component()` at precompute time.
- **Phase II-2 migration**: Once all components use `SEDComponent` Protocol, the registry will be folded into that system.
- **Radio / X-ray`: Analytic radio and X-ray models should follow the same pattern for consistency.
- **Shock lines**: MAPPINGS shock lines need a registry entry for preintegration.

## References

- `src/tengri/forward/_component_registry.py` — Registry API
- `src/tengri/components/dust/dust_emission_precompute.py` — Dust IR examples (6 models)
- `src/tengri/components/agn/skirtor_precompute.py` — SKIRTOR example (5D grid)
- `tests/unit/test_component_registry.py` — Registry tests

## See Also

- Phase II-3 Progress: `docs/dev/phase_ii3_progress.md`
- Closure Capture Design: `docs/dev/20260404-refactor.md`
- OOM Prevention: `docs/dev/notebook_orchestration_oom.md`
