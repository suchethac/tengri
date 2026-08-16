# Adding a new physics block (`SEDComponent`)

> **One-page contributor guide for the bare-`SEDComponent` Protocol —
> the advanced / fallback path.** For the recommended astronomer-facing
> authoring style (class-level Distribution attributes, `reads` /
> `publishes` dicts, single `predict` signature), see the canonical
> [`model-construction.md`](model-construction.md) and
> [`sed-model-components.md`](sed-model-components.md). For the big-picture
> architecture this fits into — `ForwardModel`, sub-models, populations,
> spatial — see
> [`forward-model-architecture.md`](archive/forward-model-architecture.md).
>
> For the migration history see
> [`phase_ii_2_stellar_migration.md`](phase_ii_2_stellar_migration.md);
> for design rationale see [`20260404-refactor.md`](20260404-refactor.md).

A new physics block is one Python file that follows the
:class:`tengri.protocols.component.SEDComponent` Protocol. Drop it in
``src/tengri/components/<domain>/component.py`` (or a sibling file if
you have multiple variants of the same domain) and it can be plugged
into a chain via :func:`tengri.forward.build_components` or hand-built
into :func:`tengri.forward.orchestrator.run_components`.

## Canonical skeleton

Copy from any shipped adapter — they are all the same shape. The
shortest reference is
[`components/radio/component.py`](../../src/tengri/components/radio/component.py);
the most complete is
[`components/dust/two_component.py`](../../src/tengri/components/dust/two_component.py).

```python
# SPDX-License-Identifier: BSD-3-Clause
"""MyPhysicsSEDComponent — one-line summary of what this block does.

Cross-component reads:
- ``state.derived["X"]`` — what upstream component publishes it.

Cross-component publications:
- ``state.derived["L_my"]`` — for downstream readers.
"""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
import jax.numpy as jnp

from tengri.protocols.component import (
    ParamDeclaration, PipelineState, SEDComponentConfig, SEDComponentState,
)
from tengri.parameters.priors import Fixed, Uniform


@dataclass(frozen=True)
class MyPhysicsSEDComponentConfig(SEDComponentConfig):
    name: str = "myphysics"
    # static knobs that don't enter the gradient tape go here.


@dataclass(frozen=True)
class MyPhysicsSEDComponentState(SEDComponentState):
    name: str = "myphysics"
    # cached static tensors go here, populated by precompute().


@dataclass(frozen=True)
class MyPhysicsSEDComponent:
    config: MyPhysicsSEDComponentConfig = field(
        default_factory=MyPhysicsSEDComponentConfig
    )
    name: str = "myphysics"
    parameter_prefix: str = "my_"  # CI-enforced (tools/check_param_prefixes.py).

    def declared_parameters(self) -> list[ParamDeclaration]:
        # Mirror entries in tengri.parameters._param_defs — do not duplicate.
        return [
            ParamDeclaration("my_knob", Fixed(0.0), "What it does [units]"),
        ]

    def precompute(
        self,
        ssp_data: Any | None = None,
        wave_grid: jnp.ndarray | None = None,
    ) -> MyPhysicsSEDComponentState:
        # Eager (non-JIT) work: load HDF5 grids, build static tensors.
        # Return a marker if your block has no precomputed tensors.
        del ssp_data, wave_grid
        return MyPhysicsSEDComponentState(name=self.name)

    def apply(
        self,
        state: PipelineState,
        params: Mapping[str, jnp.ndarray],
    ) -> PipelineState:
        # Pure JAX. No file I/O, no numpy primitives, no float() casts on
        # traced arrays.
        new_sed = ...  # whatever your physics computes
        # Typed bundle write (ADR-0007 Phase 3+). The bundle's with_
        # raises TypeError on unknown fields, so typos in derived keys
        # fail at trace time rather than at the consumer's read site.
        return state.with_(
            sed_intrinsic=new_sed,
            derived=state.derived.with_(L_my=...),
        )
```

## Five rules

1. **Parameters live in `_param_defs.py`.** Your `declared_parameters`
   returns priors from the existing registry — never duplicate them.
   If a parameter doesn't exist there, add it there in the same PR.
2. **Cross-component data flows through `state.derived`.** Never reach
   into another component's `params` or attributes. Document the keys
   you read with their fallback (e.g.
   `state.derived.get("L_ir", 0.0)`) and the keys you publish.
3. **`precompute()` is eager. `apply()` is pure JAX.** File I/O,
   HDF5 loads, registry lookups happen in `precompute`. Inside `apply`
   you must not call `float()` on a traced parameter; use `jnp.asarray`.
4. **The component itself is a frozen dataclass.** Hold static tensors
   on the component (parallel to `config`) for the natural plumbing,
   or return them from `precompute()` if your downstream consumer
   wants to manage the cache. ``StellarSEDComponent`` chooses the
   former; the rest are stateless.
5. **No string values in `state.derived`.** Strings are not JAX leaves
   and break `jax.jit`. Use numeric flags or document-only attributes
   on `self.config`.

## Wiring into a pipeline

```python
from tengri.forward import build_components, run_components
from tengri.protocols.component import PipelineState

components = build_components(
    ssp_data=ssp,
    sfh_model="tsnorm", metallicity_model="ramp",
    nebular_backend="cue", nebular_backend_instance=cue,
    agn_model="standard",
    dust_law_bc="calzetti", dust_emission_model="dale2014",
    use_radio=True, use_xray=True, use_igm=True,
)
state = run_components(
    components,
    PipelineState(wave=ssp.ssp_wave),
    params,
)
# state.sed_intrinsic, state.derived[<your keys>] etc. populated.
```

`run_components` JIT-compiles cleanly: wrap it in `jax.jit` and the
chain produces bit-exact identical results to the eager path. See
[`tests/integration/test_orchestrator_jit.py`](../../tests/integration/test_orchestrator_jit.py)
for parametrized verification.

## Gotchas

- **`agn_torus_frac`**: do NOT auto-derive from `cos(theta_torus)` —
  gradient discontinuity. Independent free parameter.
- **PSD timescale**: user-facing `psd_tau_myr` (Myr); internal is yr
  (×1e6 conversion in the registry's `internal_param_map`).
- **Metallicity**: `met_logzsol` is log10(Z/Zsun); the absolute log10(Z)
  used by SSP grids is `met_logzsol + LOG10_ZSUN`.
- **JAX-Metal**: Apple-GPU JAX produces NaNs in some DSPS paths.
  Tests should set `JAX_PLATFORMS=cpu`.

## Where to look next

- [`src/tengri/core/component.py`](../../src/tengri/core/component.py)
  — the contract and the JAX-pytree registration.
- [`src/tengri/forward/orchestrator.py`](../../src/tengri/forward/orchestrator.py)
  — `slice_params_for_component`, `merge_declared_parameters`,
  `run_components`, `sample_params_dict`, `default_params_dict`.
- [`src/tengri/forward/component_factory.py`](../../src/tengri/forward/component_factory.py)
  — public-API factory.
- [`src/tengri/parameters/_param_defs.py`](../../src/tengri/parameters/_param_defs.py)
  — the parameter registry every component pulls priors from.
