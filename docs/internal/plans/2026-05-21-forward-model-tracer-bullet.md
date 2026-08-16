# ForwardModel Tracer-Bullet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the outer `ForwardModel` shell, a `Population` dataclass, and a `SubModel` Protocol as a thin, purely additive layer over the existing `SEDModel`. End state: a user calls `ForwardModel.build(sed=existing_sed_model, observation=existing_obs).predict(params)` and gets back a prediction dict suitable for inference, with the same numerical result as today and with the per-population orchestration structure (§9.1 of the architecture spec) in place — even though there is only one population in this slice.

**Architecture:** Additive shell. No existing class is renamed, deleted, or restructured. The new `ForwardModel` wraps existing objects via a `SubModel`-conforming adapter around `SEDModel`. The persistent JIT cache rule (§9.2 closure minimization) is audited by an integration test that exercises the new shell under `@jax.jit`. `SpatialModel`, `SpatialSEDModel`, multi-population, and the parameter-namespace prefix all land in subsequent plans.

**Tech Stack:** Python 3.12, JAX ≥0.4.20, pytest, ruff (lint+format), numpydoc docstrings. Project conventions in `CLAUDE.md`.

**Out of scope (subsequent plans):**
- `SpatialComponent` Protocol, `SpatialModel`, `SpatialModelComponent` astronomer-facing base
- `SpatialSEDModel`
- Multi-population (`populations: tuple[Population, ...]`) — the slice ships single-pop only
- The `<population>.<prefix>_<param>` namespace and CI guard rewrite (ADR-0012 §6.1)
- Factoring SEDModel's outer shell out of `forward/sed_model.py`
- New observation adapters (`FiberSpectroscopyObservation`, `ImagingObservation`)

**Pre-flight check (run before Task 1):**

```bash
.venv/bin/pytest tests/ -q --co 2>&1 | tail -3
.venv/bin/ruff check src/ tests/
```

Both should pass. If not, fix or surface to the user before starting Task 1.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/tengri/protocols/submodel.py` | Create | `SubModel` Protocol — 2 methods: `declared_parameters()`, `run(state, params)`. |
| `src/tengri/protocols/__init__.py` | Modify | Re-export `SubModel`. |
| `src/tengri/forward/population.py` | Create | Frozen `Population` dataclass: `name`, `sed: SubModel`, `spatial: SubModel | None`. |
| `src/tengri/forward/forward_model.py` | Create | `ForwardModel` class + `build()` classmethod + `predict()` method. Holds `populations: tuple[Population, ...]` and `observation`. Single-pop only in this slice. |
| `src/tengri/forward/_sed_submodel_adapter.py` | Create | `_LegacySEDSubModel` — wraps existing `SEDModel` to satisfy `SubModel` Protocol. Underscore-prefix: an explicit migration shim, not public API. |
| `src/tengri/__init__.py` | Modify | Public re-exports of `ForwardModel`, `Population`. |
| `tests/unit/protocols/test_submodel_protocol.py` | Create | Protocol shape + runtime-checkable assertions. |
| `tests/unit/forward/test_population.py` | Create | `Population` dataclass behavior. |
| `tests/unit/forward/test_forward_model.py` | Create | `ForwardModel.build` kwargs convenience + `.predict` dict shape + per-population structure. |
| `tests/integration/test_forward_model_jit.py` | Create | End-to-end JIT works through the shell + closure audit (§9.2). |
| `docs/dev/where-things-live.md` | Modify | New entry for `ForwardModel` under "Forward model". |
| `CHANGELOG.md` | Modify | Single entry under Unreleased. |

Total new files: 7. Modified files: 4. No file is deleted or renamed in this slice.

---

## Task 1: SubModel Protocol

**Files:**
- Create: `src/tengri/protocols/submodel.py`
- Test: `tests/unit/protocols/test_submodel_protocol.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/protocols/test_submodel_protocol.py`:

```python
"""Smoke tests for the SubModel Protocol (forward-model architecture §4)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.protocols import SubModel
from tengri.protocols.component import ForwardState, ParamDeclaration


class _MinimalSubModel:
    """Smallest possible SubModel implementation, for shape checks only."""

    name = "minimal"

    def declared_parameters(self) -> list[ParamDeclaration]:
        return []

    def run(self, state: ForwardState, params) -> ForwardState:
        return state


def test_submodel_is_runtime_checkable() -> None:
    assert isinstance(_MinimalSubModel(), SubModel)


def test_submodel_rejects_missing_run() -> None:
    class Broken:
        name = "broken"

        def declared_parameters(self) -> list[ParamDeclaration]:
            return []

    assert not isinstance(Broken(), SubModel)


def test_submodel_rejects_missing_declared_parameters() -> None:
    class Broken:
        name = "broken"

        def run(self, state, params):
            return state

    assert not isinstance(Broken(), SubModel)


def test_submodel_minimal_run_returns_state() -> None:
    sub = _MinimalSubModel()
    state = ForwardState(wave=jnp.array([1000.0, 2000.0, 3000.0]))
    out = sub.run(state, {})
    assert out is state
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/protocols/test_submodel_protocol.py -v
```

Expected: collection error or `ImportError: cannot import name 'SubModel'`.

- [ ] **Step 3: Write the Protocol**

Create `src/tengri/protocols/submodel.py`:

```python
"""SubModel protocol: one mode of the ForwardModel (SED, spatial, joint).

Defined by the forward-model architecture spec
(``docs/dev/archive/forward-model-architecture.md``) §4. A SubModel is the thin
composer over a list of components. Each population carries one SED
SubModel and optionally one spatial SubModel; ``ForwardModel`` runs the
populations in sequence and hands the result to ``ObservationModel``.

This is a Protocol, not an ABC — implementations satisfy it by shape.
The runtime-checkable variant is provided so smoke tests can assert
``isinstance(obj, SubModel)`` without importing a concrete base.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp

from tengri.protocols.component import ForwardState

__all__ = ["SubModel"]


@runtime_checkable
class SubModel(Protocol):
    """Contract for one mode of a :class:`ForwardModel`.

    Required attributes
    -------------------
    name : str
        Stable identifier for diagnostics. Examples: ``"sed"``,
        ``"spatial"``, ``"spatial_sed"``.

    Required methods
    ----------------
    declared_parameters() -> list[ParamDeclaration]
        Aggregated parameter declarations across this SubModel's
        components. Consumed by ``Parameters`` and by the CI prefix
        guard.

    run(state, params) -> ForwardState
        Pure JAX. Threads the input state through the SubModel's
        components and returns a new ``ForwardState`` with the
        SubModel's contribution applied. Must not mutate the input.

    Notes
    -----
    JIT/grad/vmap-compatible: ``run`` is pure JAX. Static configuration
    (component list, frozen pytree state) is held on ``self`` as
    Python attributes, not captured by closure.
    """

    name: str

    def declared_parameters(self) -> list[Any]:
        """Aggregated parameter declarations from this SubModel's components."""
        ...

    def run(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> ForwardState:
        """Pure JAX. Thread state through this SubModel's components."""
        ...
```

- [ ] **Step 4: Wire export**

Edit `src/tengri/protocols/__init__.py`. Find the existing imports block and add:

```python
from tengri.protocols.submodel import SubModel
```

Add `"SubModel"` to the `__all__` list (keep alphabetical order).

- [ ] **Step 5: Run test to verify it passes**

```bash
.venv/bin/pytest tests/unit/protocols/test_submodel_protocol.py -v
```

Expected: 4 passed.

- [ ] **Step 6: Lint**

```bash
.venv/bin/ruff check src/tengri/protocols/ tests/unit/protocols/
.venv/bin/ruff format --check src/tengri/protocols/ tests/unit/protocols/
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/tengri/protocols/submodel.py src/tengri/protocols/__init__.py tests/unit/protocols/test_submodel_protocol.py
git commit -m "feat(protocols): SubModel Protocol (forward-model architecture §4)"
```

---

## Task 2: Population dataclass

**Files:**
- Create: `src/tengri/forward/population.py`
- Test: `tests/unit/forward/test_population.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/forward/test_population.py`:

```python
"""Tests for the Population dataclass (forward-model architecture §5)."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import pytest

from tengri.forward.population import Population
from tengri.protocols.component import ForwardState


class _DummySubModel:
    name = "dummy"

    def declared_parameters(self):
        return []

    def run(self, state, params):
        return state


def test_population_is_frozen() -> None:
    pop = Population(name="default", sed=_DummySubModel(), spatial=None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        pop.name = "other"  # type: ignore[misc]


def test_population_holds_sed_and_optional_spatial() -> None:
    sub = _DummySubModel()
    pop = Population(name="bulge", sed=sub, spatial=None)
    assert pop.name == "bulge"
    assert pop.sed is sub
    assert pop.spatial is None


def test_population_spatial_defaults_to_none() -> None:
    pop = Population(name="x", sed=_DummySubModel())
    assert pop.spatial is None


def test_population_name_must_be_nonempty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Population(name="", sed=_DummySubModel())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/forward/test_population.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

Create `src/tengri/forward/population.py`:

```python
"""Population — one (SED, spatial) pair inside a :class:`ForwardModel`.

A galaxy decomposition (AGN point source + Sérsic bulge + exponential
disc) is expressed as multiple :class:`Population`s. The tracer-bullet
implementation supports single-population only; ADR-0012 will lift
this to multi-population with namespaced parameter names.

See ``docs/dev/archive/forward-model-architecture.md`` §5 and ADR-0012.
"""

from __future__ import annotations

from dataclasses import dataclass

from tengri.protocols.submodel import SubModel

__all__ = ["Population"]


@dataclass(frozen=True)
class Population:
    """One (SED, spatial) pair inside a :class:`ForwardModel`.

    Parameters
    ----------
    name : str
        Population namespace. ``"default"`` for the convenience
        single-population path. Used by ADR-0012 multi-population
        parameter naming (not yet active in this slice).
    sed : SubModel
        SED SubModel for this population.
    spatial : SubModel or None, optional
        Spatial SubModel for this population. ``None`` for SED-only
        populations.
    """

    name: str
    sed: SubModel
    spatial: SubModel | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Population.name must be a non-empty string.")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/unit/forward/test_population.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Lint**

```bash
.venv/bin/ruff check src/tengri/forward/population.py tests/unit/forward/test_population.py
.venv/bin/ruff format --check src/tengri/forward/population.py tests/unit/forward/test_population.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/tengri/forward/population.py tests/unit/forward/test_population.py
git commit -m "feat(forward): Population dataclass (forward-model architecture §5)"
```

---

## Task 3: Legacy SEDModel → SubModel adapter

**Files:**
- Create: `src/tengri/forward/_sed_submodel_adapter.py`
- Test: `tests/unit/forward/test_sed_submodel_adapter.py`

This is the migration shim. It exists *only* until the next plan factors SEDModel's chain into a first-class SubModel. The underscore prefix flags that it is not public API.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/forward/test_sed_submodel_adapter.py`:

```python
"""Tests for _LegacySEDSubModel — wraps existing SEDModel as a SubModel."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.forward._sed_submodel_adapter import _LegacySEDSubModel
from tengri.protocols import SubModel


@pytest.fixture
def sed_model_smallest(ssp_data, photometry_obs_smallest):
    """A minimal SEDModel for shape testing. Reuses existing test fixtures."""
    from tengri import SEDModel, FIXED, FREE

    return SEDModel.build(
        ssp_data=ssp_data,
        observation=photometry_obs_smallest,
        sfh={"type": "dpl", "*": FIXED},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "off"},
    )


def test_adapter_satisfies_submodel_protocol(sed_model_smallest) -> None:
    sub = _LegacySEDSubModel(sed_model_smallest)
    assert isinstance(sub, SubModel)


def test_adapter_name_is_sed(sed_model_smallest) -> None:
    sub = _LegacySEDSubModel(sed_model_smallest)
    assert sub.name == "sed"


def test_adapter_declared_parameters_delegates(sed_model_smallest) -> None:
    sub = _LegacySEDSubModel(sed_model_smallest)
    declared = sub.declared_parameters()
    # Free params should equal what the wrapped SEDModel declares.
    assert {d.name for d in declared} == set(sed_model_smallest.spec.free_params)


def test_adapter_holds_sed_by_reference(sed_model_smallest) -> None:
    sub = _LegacySEDSubModel(sed_model_smallest)
    assert sub.sed_model is sed_model_smallest
```

> **Fixture note for the implementer:** `ssp_data` and `photometry_obs_smallest` fixtures must exist in `tests/conftest.py` or a closer-scoped `conftest.py`. If they do not, create a `tests/unit/forward/conftest.py` that constructs minimal versions. Reuse existing test fixtures wherever possible; do not invent a new SSP loader.

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/forward/test_sed_submodel_adapter.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

Create `src/tengri/forward/_sed_submodel_adapter.py`:

```python
"""_LegacySEDSubModel — migration shim wrapping :class:`SEDModel` as a :class:`SubModel`.

This adapter exists to land the ``ForwardModel`` outer shell without
disturbing the existing 2957-line :class:`tengri.forward.sed_model.SEDModel`
internals. Subsequent plans factor SEDModel's chain into a first-class
SubModel; at that point this adapter is deleted.

See the tracer-bullet plan at
``docs/internal/plans/2026-05-21-forward-model-tracer-bullet.md``.

Underscore-prefixed because it is not part of the public API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from tengri.protocols.component import ForwardState, ParamDeclaration

if TYPE_CHECKING:
    from tengri.forward.sed_model import SEDModel

__all__ = ["_LegacySEDSubModel"]


@dataclass(frozen=True)
class _LegacySEDSubModel:
    """SubModel adapter around an existing :class:`SEDModel`.

    Parameters
    ----------
    sed_model : SEDModel
        Constructed SEDModel instance. Held by reference; not copied.

    Notes
    -----
    The adapter does no physics of its own. ``run(state, params)``
    delegates to the wrapped SEDModel's internal forward pipeline.
    ``declared_parameters`` re-shapes ``sed_model.spec`` into a list
    of :class:`ParamDeclaration` so :class:`ForwardModel` can
    introspect free parameters uniformly across populations.
    """

    sed_model: "SEDModel"
    name: str = "sed"

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameter declarations from the wrapped SEDModel.

        Returns the same set as ``sed_model.spec.free_params``, lifted
        into :class:`ParamDeclaration` tuples so the result type matches
        the SubModel Protocol.
        """
        spec = self.sed_model.spec
        return [
            ParamDeclaration(
                name=name,
                prior=spec.prior_of(name),
                description="",
                units="",
            )
            for name in spec.free_params
        ]

    def run(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> ForwardState:
        """Run the wrapped SEDModel's forward pipeline.

        Reuses ``SEDModel._run_components`` (the orchestrator entry
        point that produces a ``ForwardState``). The shape matches:
        existing pipeline returns a ``ForwardState``-equivalent, the
        adapter passes it through unchanged.

        Parameters
        ----------
        state : ForwardState
            Input state (typically empty: ``ForwardState(wave=...)``).
        params : mapping
            Free parameter values.

        Returns
        -------
        ForwardState
            New state with SED contributions applied. Pure JAX.
        """
        return self.sed_model._run_components(state, params)
```

> **Implementer check before commit:** Confirm `SEDModel._run_components(state, params)` exists with this signature. If not, the closest equivalent is `SEDModel.predict_observables_jit` or the orchestrator entry point. Find the right method by:
>
> ```bash
> grep -n "_run_components\|run_components\|predict_observables_jit" src/tengri/forward/sed_model.py | head
> ```
>
> Use the method that takes `(state, params)` and returns a `ForwardState`. If neither exists in that exact shape, the adapter's `run` method builds the equivalent by calling the orchestrator directly:
>
> ```python
> from tengri.forward.orchestrator import run_components
>
> def run(self, state, params):
>     return run_components(self.sed_model._components, state, params, ssp_data=self.sed_model.ssp_data)
> ```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/unit/forward/test_sed_submodel_adapter.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Lint**

```bash
.venv/bin/ruff check src/tengri/forward/_sed_submodel_adapter.py tests/unit/forward/test_sed_submodel_adapter.py
.venv/bin/ruff format --check src/tengri/forward/_sed_submodel_adapter.py tests/unit/forward/test_sed_submodel_adapter.py
```

- [ ] **Step 6: Commit**

```bash
git add src/tengri/forward/_sed_submodel_adapter.py tests/unit/forward/test_sed_submodel_adapter.py
git commit -m "feat(forward): _LegacySEDSubModel adapter wrapping existing SEDModel"
```

---

## Task 4: ForwardModel — construction and `.build()` kwargs convenience

**Files:**
- Create: `src/tengri/forward/forward_model.py`
- Test: `tests/unit/forward/test_forward_model.py` (Step 1 only; further steps in Task 5)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/forward/test_forward_model.py`:

```python
"""Tests for ForwardModel (forward-model architecture §5).

Tracer-bullet scope: single-population only. Multi-population lives
in the ADR-0012 plan.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.forward.forward_model import ForwardModel
from tengri.forward.population import Population


def test_build_single_population_from_sed_kwarg(sed_model_smallest, photometry_obs_smallest) -> None:
    forward = ForwardModel.build(
        sed=sed_model_smallest,
        observation=photometry_obs_smallest,
    )
    assert isinstance(forward, ForwardModel)
    assert len(forward.populations) == 1
    assert forward.populations[0].name == "default"


def test_build_rejects_no_sed_no_populations(photometry_obs_smallest) -> None:
    with pytest.raises(ValueError, match="sed=.*or.*populations="):
        ForwardModel.build(observation=photometry_obs_smallest)


def test_build_accepts_explicit_populations(sed_model_smallest, photometry_obs_smallest) -> None:
    from tengri.forward._sed_submodel_adapter import _LegacySEDSubModel

    pop = Population(name="only", sed=_LegacySEDSubModel(sed_model_smallest))
    forward = ForwardModel.build(populations=[pop], observation=photometry_obs_smallest)
    assert forward.populations[0].name == "only"


def test_build_rejects_multi_population_in_tracer_bullet(sed_model_smallest, photometry_obs_smallest) -> None:
    """Multi-population is deferred to ADR-0012 plan. Tracer-bullet ships single-pop."""
    from tengri.forward._sed_submodel_adapter import _LegacySEDSubModel

    pops = [
        Population(name="a", sed=_LegacySEDSubModel(sed_model_smallest)),
        Population(name="b", sed=_LegacySEDSubModel(sed_model_smallest)),
    ]
    with pytest.raises(NotImplementedError, match="ADR-0012"):
        ForwardModel.build(populations=pops, observation=photometry_obs_smallest)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/forward/test_forward_model.py::test_build_single_population_from_sed_kwarg -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

Create `src/tengri/forward/forward_model.py`:

```python
"""ForwardModel — the outer shell of the forward chain.

Owns a tuple of :class:`Population`s and an :class:`Observation`.
Exposes a single ``.predict(params)`` method that inference calls.
The architecture spec is at
``docs/dev/archive/forward-model-architecture.md`` §5; this file implements
the tracer-bullet single-population slice.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.forward._sed_submodel_adapter import _LegacySEDSubModel
from tengri.forward.population import Population
from tengri.protocols.component import ForwardState

__all__ = ["ForwardModel"]


@dataclass(frozen=True)
class ForwardModel:
    """The outer shell of the forward model.

    Holds populations + observation; exposes ``.predict(params)`` as the
    sole API inference consumes. See
    ``docs/dev/archive/forward-model-architecture.md`` for the full design.

    Tracer-bullet limitations (this slice):
      * Single population only.
      * Spatial SubModels are not constructed (subsequent plan).
      * Parameter names are not namespaced (subsequent plan, ADR-0012).

    Parameters
    ----------
    populations : tuple of Population
        At least one population; tracer-bullet enforces exactly one.
    observation : object
        Observation model — anything with a ``.predict(state, params)``
        method that returns a mapping of channel keys to arrays.
        Concrete types: :class:`tengri.observation.Observation` and the
        Protocol-shaped photometry / spectroscopy / joint adapters.
    """

    populations: tuple[Population, ...]
    observation: Any

    @classmethod
    def build(
        cls,
        *,
        sed: Any | None = None,
        spatial: Any | None = None,
        populations: Iterable[Population] | None = None,
        observation: Any,
    ) -> "ForwardModel":
        """Construct a :class:`ForwardModel`.

        Convenience entry point. Two forms:

        - **Single-population sugar (the common case):** pass
          ``sed=<SEDModel>`` and ``observation=<Observation>``. The
          SED is wrapped into a one-element ``populations`` tuple with
          ``name="default"``.
        - **Explicit populations:** pass ``populations=[...]``. Used
          once multi-population lands (ADR-0012). The tracer-bullet
          accepts the form but raises on >1 entry.

        Parameters
        ----------
        sed : SEDModel or SubModel, optional
            Single-population shortcut. Mutually exclusive with
            ``populations``.
        spatial : optional
            Reserved for the spatial plan. Raises if provided in the
            tracer-bullet.
        populations : iterable of Population, optional
            Explicit population list. Mutually exclusive with ``sed``.
        observation : object
            Observation model.

        Returns
        -------
        ForwardModel

        Raises
        ------
        ValueError
            If neither ``sed`` nor ``populations`` is given, or both.
        NotImplementedError
            If ``len(populations) > 1`` (multi-population is ADR-0012)
            or if ``spatial`` is provided (subsequent plan).
        """
        if spatial is not None:
            raise NotImplementedError(
                "ForwardModel.build(spatial=...) is reserved for the spatial-model plan."
            )
        if (sed is None) == (populations is None):
            raise ValueError(
                "ForwardModel.build needs exactly one of sed=... or populations=..."
            )

        if sed is not None:
            from tengri.forward.sed_model import SEDModel  # local import to avoid cycle

            sub = sed if not isinstance(sed, SEDModel) else _LegacySEDSubModel(sed)
            pops = (Population(name="default", sed=sub),)
        else:
            assert populations is not None
            pops = tuple(populations)

        if len(pops) > 1:
            raise NotImplementedError(
                "Multi-population ForwardModel is deferred to the ADR-0012 plan. "
                "This tracer-bullet ships single-population only."
            )
        if len(pops) == 0:
            raise ValueError("ForwardModel needs at least one population.")

        return cls(populations=pops, observation=observation)

    def predict(self, params: Mapping[str, jnp.ndarray]) -> Mapping[str, jnp.ndarray]:
        """Predicted observables for the given parameters.

        Iterates over ``populations`` (per-population orchestration —
        architecture spec §9.1), produces per-population
        :class:`ForwardState`s, hands them to the observation, returns
        the prediction dict.

        Single-population case: the loop is a one-element loop;
        the resulting per-population dict has one key (``"default"``)
        which the observation's ``predict_summed`` reduces trivially.

        Parameters
        ----------
        params : mapping of str -> array
            Free parameter values.

        Returns
        -------
        mapping of str -> array
            Prediction dict. Keys depend on the observation
            (``"phot_fnu"``, ``"spec_fnu"``, …).
        """
        wave = self.observation.wave_grid_rest_aa
        per_pop_states: dict[str, ForwardState] = {}
        for pop in self.populations:
            state = ForwardState(wave=wave)
            state = pop.sed.run(state, params)
            if pop.spatial is not None:
                state = pop.spatial.run(state, params)
            per_pop_states[pop.name] = state

        # Single-pop fast path: hand the only state to the observation.
        # Multi-population summing will be added in the ADR-0012 plan
        # (linear-flux sum at the observation layer).
        if len(per_pop_states) == 1:
            (only_state,) = per_pop_states.values()
            return self.observation.predict(only_state, params)
        raise NotImplementedError(
            "Multi-population summing is the ADR-0012 plan, not this slice."
        )
```

> **Implementer check before commit:** `self.observation.wave_grid_rest_aa` is the assumed wavelength attribute on existing observation objects. If the actual attribute name differs, replace with the correct one — search:
>
> ```bash
> grep -n "wave_grid_rest_aa\|wave_rest_aa\|wave_grid\|_wave" src/tengri/observation/observation.py src/tengri/observation/photometry_model.py | head
> ```
>
> Pick the rest-frame Angstrom grid the existing predict path uses. If observation objects do not expose one, the adapter's `_LegacySEDSubModel.run` can take an empty state — adapt by passing `state = ForwardState(wave=self.populations[0].sed.sed_model._wave_rest_aa)` if necessary. Whatever works for the existing forward path works here.

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/unit/forward/test_forward_model.py -v -k "build"
```

Expected: 4 passed (the 4 `test_build_*` tests).

- [ ] **Step 5: Lint**

```bash
.venv/bin/ruff check src/tengri/forward/forward_model.py tests/unit/forward/test_forward_model.py
.venv/bin/ruff format --check src/tengri/forward/forward_model.py tests/unit/forward/test_forward_model.py
```

- [ ] **Step 6: Commit**

```bash
git add src/tengri/forward/forward_model.py tests/unit/forward/test_forward_model.py
git commit -m "feat(forward): ForwardModel.build single-population kwargs convenience"
```

---

## Task 5: ForwardModel.predict — full prediction dict round-trip

**Files:**
- Modify: `tests/unit/forward/test_forward_model.py` (add `predict` tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/forward/test_forward_model.py`:

```python
def test_predict_returns_mapping_with_expected_keys(sed_model_smallest, photometry_obs_smallest) -> None:
    forward = ForwardModel.build(sed=sed_model_smallest, observation=photometry_obs_smallest)
    params = sed_model_smallest.spec.fiducial_dict()
    pred = forward.predict(params)
    assert isinstance(pred, dict)
    # The existing predict path produces at least one of these keys.
    assert any(k in pred for k in ("phot_fnu", "spec_fnu", "lines_flux"))


def test_predict_matches_legacy_sedmodel(sed_model_smallest, photometry_obs_smallest) -> None:
    """The shell must not change the numerical result vs the existing path."""
    import jax.numpy as jnp

    forward = ForwardModel.build(sed=sed_model_smallest, observation=photometry_obs_smallest)
    params = sed_model_smallest.spec.fiducial_dict()
    pred_new = forward.predict(params)
    pred_old = sed_model_smallest.predict_photometry(params)

    # Match whatever key the photometric model uses.
    new_phot = pred_new.get("phot_fnu", pred_new.get("fnu_obs"))
    assert new_phot is not None, f"Prediction dict missing photometric key: {list(pred_new)}"
    assert jnp.allclose(new_phot, pred_old, rtol=1e-10, atol=0.0)
```

> **Implementer check:** `sed_model_smallest.spec.fiducial_dict()` is the prior-fiducial-values dict. If the spec method has a different name (e.g. `init_values`, `defaults`), substitute. Same for `predict_photometry` vs `predict_phot` — match what `SEDModel` actually exposes.

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/unit/forward/test_forward_model.py::test_predict_returns_mapping_with_expected_keys -v
```

Expected: failure (probably `AttributeError` on a method name; resolve by checking the legacy `SEDModel` interface).

- [ ] **Step 3: Fix the implementation to match**

The implementation in Task 4 used `self.observation.predict(state, params)`. If the actual observation object does not have a `predict(state, params)` method but instead has a flatter API (e.g. `project(state)`, `predict_fnu(state, params)`), adapt the line in `forward_model.py`'s `predict` method.

Search the existing forward chain to see what's called:

```bash
grep -n "observation\.\(predict\|project\|run\)" src/tengri/forward/sed_model.py | head
```

If the existing call is, for example, `obs.project(state)`, replace `self.observation.predict(only_state, params)` in `forward_model.py` with `self.observation.project(only_state, params)` (or whatever shape matches). Keep this slice mechanical — do not redesign the observation API.

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/unit/forward/test_forward_model.py -v
```

Expected: 6 passed (4 build + 2 predict).

- [ ] **Step 5: Commit**

```bash
git add src/tengri/forward/forward_model.py tests/unit/forward/test_forward_model.py
git commit -m "feat(forward): ForwardModel.predict numerically equivalent to legacy SEDModel"
```

---

## Task 6: Integration test — end-to-end JIT through ForwardModel

**Files:**
- Create: `tests/integration/test_forward_model_jit.py`

This test exercises the architecture spec §9.1 "End-to-end JIT" mode: a user wraps `forward.predict` in `@jax.jit` and computes a loss. The new shell must be transparent under tracing.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_forward_model_jit.py`:

```python
"""End-to-end JIT through ForwardModel (forward-model architecture §9.1).

The new shell must be transparent under jax.jit. A loss function that
wraps forward.predict + a likelihood computation must JIT, run, and
take gradients without error.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


@pytest.mark.integration
def test_forward_model_end_to_end_jit(sed_model_smallest, photometry_obs_smallest) -> None:
    from tengri.forward.forward_model import ForwardModel

    forward = ForwardModel.build(sed=sed_model_smallest, observation=photometry_obs_smallest)

    @jax.jit
    def loss(params):
        pred = forward.predict(params)
        # Reuse whatever photometry key the observation publishes.
        for key in ("phot_fnu", "fnu_obs"):
            if key in pred:
                return jnp.sum(pred[key] ** 2)
        raise AssertionError(f"Prediction dict missing photometric key: {list(pred)}")

    params = sed_model_smallest.spec.fiducial_dict()
    value = loss(params)
    assert jnp.isfinite(value)

    # Gradient must trace too.
    grads = jax.grad(loss)(params)
    assert all(jnp.isfinite(g).all() for g in grads.values())


@pytest.mark.integration
def test_forward_model_closure_audit(sed_model_smallest, photometry_obs_smallest) -> None:
    """ForwardModel.predict must not capture data-file globals (§9.2)."""
    from tengri.forward.forward_model import ForwardModel

    forward = ForwardModel.build(sed=sed_model_smallest, observation=photometry_obs_smallest)
    closure = forward.predict.__func__.__closure__ or ()
    # The bound method's free variables should be at most 0; predict only
    # closes over `self`, which is the ``ForwardModel`` instance and an
    # explicit argument under JIT.
    assert len(closure) == 0, (
        f"ForwardModel.predict closes over {len(closure)} free variables; "
        "data files must flow through component state, not closures (§9.2)."
    )
```

- [ ] **Step 2: Run test to verify it fails (or passes — see below)**

```bash
.venv/bin/pytest tests/integration/test_forward_model_jit.py -v
```

Two outcomes are acceptable here:

- **Both pass.** Excellent — the shell is JIT-clean by construction. Continue to Step 3.
- **`test_forward_model_end_to_end_jit` fails with a tracing error.** Likely cause: the `_LegacySEDSubModel.run` adapter calls into a non-JIT-clean path inside SEDModel. Diagnose with `pytest --tb=long`, identify the leaf call, and either:
   - Route the adapter through the JIT-clean orchestrator entry point (see Task 3 implementer note), or
   - If no JIT-clean entry exists, mark this test `@pytest.mark.xfail(reason="legacy SEDModel.run is not JIT-clean; tracked for follow-up")` and proceed.

Adapt rather than redesign — the architecture refactor of `SEDModel`'s internals is out of scope for this slice.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_forward_model_jit.py
git commit -m "test(forward): end-to-end JIT and closure audit through ForwardModel shell"
```

---

## Task 7: Public exports

**Files:**
- Modify: `src/tengri/__init__.py`

- [ ] **Step 1: Add the exports**

Edit `src/tengri/__init__.py`. Find the existing public-symbol block (search for `__all__` or the existing `SEDModel` import). Add:

```python
from tengri.forward.forward_model import ForwardModel
from tengri.forward.population import Population
```

Add `"ForwardModel"` and `"Population"` to `__all__` (alphabetical).

- [ ] **Step 2: Verify exports**

```bash
.venv/bin/python -c "from tengri import ForwardModel, Population; print(ForwardModel, Population)"
```

Expected: two class names printed.

- [ ] **Step 3: Lint**

```bash
.venv/bin/ruff check src/tengri/__init__.py
.venv/bin/ruff format --check src/tengri/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add src/tengri/__init__.py
git commit -m "feat(tengri): export ForwardModel and Population at top level"
```

---

## Task 8: Documentation — where-things-live + CHANGELOG

**Files:**
- Modify: `docs/dev/where-things-live.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update where-things-live**

Edit `docs/dev/where-things-live.md`. Find the section that lists `forward/` symbols (search for `sed_model.py`). Add an entry immediately under `SEDModel`:

```markdown
| Forward model outer shell | `src/tengri/forward/forward_model.py` | `ForwardModel`, `ForwardModel.build`, `ForwardModel.predict`. The class inference talks to. Wraps populations + observation. Tracer-bullet: single-population only; multi-population is ADR-0012. |
| Population dataclass | `src/tengri/forward/population.py` | `Population(name, sed, spatial)`. One (SED, spatial) pair held by `ForwardModel`. |
| SubModel Protocol | `src/tengri/protocols/submodel.py` | `SubModel` — runtime-checkable contract for one mode of `ForwardModel` (`run`, `declared_parameters`). |
```

- [ ] **Step 2: Update CHANGELOG**

Edit `CHANGELOG.md`. Under the `## Unreleased` section, add (alphabetized under the right subhead):

```markdown
### Added

- **`tengri.ForwardModel`** — the outer-shell forward-model class.
  Wraps populations + observation and exposes a single
  ``.predict(params)`` method. Tracer-bullet single-population only;
  multi-population lands in the ADR-0012 follow-up plan. See
  ``docs/dev/archive/forward-model-architecture.md``.
- **`tengri.Population`** — one (SED, spatial) pair held by
  ``ForwardModel``. Spatial submodel is reserved (``None``) in this
  slice.
- **`tengri.protocols.SubModel`** — runtime-checkable Protocol for
  one mode of ``ForwardModel`` (SED, spatial, joint).
```

- [ ] **Step 3: Commit**

```bash
git add docs/dev/where-things-live.md CHANGELOG.md
git commit -m "docs: ForwardModel + Population + SubModel entries"
```

---

## Task 9: Full-suite regression check

This task has no new code — it confirms the tracer-bullet did not break anything else.

- [ ] **Step 1: Run unit tests**

```bash
.venv/bin/pytest tests/unit/ -q
```

Expected: all pass. If anything fails that is not in `tests/unit/forward/` or `tests/unit/protocols/`, the failure is regression caused by this slice — diagnose with `pytest --tb=short` before proceeding.

- [ ] **Step 2: Run integration tests (skips integration tests that require missing SSP data files)**

```bash
.venv/bin/pytest tests/integration/ -q
```

Expected: all pass *or* skip with the documented "needs SSP data" reason.

- [ ] **Step 3: Lint and format full project**

```bash
.venv/bin/ruff check src/ tests/
.venv/bin/ruff format --check src/ tests/
```

Expected: clean. If not, run `.venv/bin/ruff check --fix src/ tests/` and `.venv/bin/ruff format src/ tests/`, then re-run the checks and commit any auto-fixes:

```bash
git add -u
git commit -m "chore: ruff auto-fixes after ForwardModel tracer-bullet"
```

- [ ] **Step 4: Smoke-test public import**

```bash
.venv/bin/python -c "
import tengri
print('ForwardModel:', tengri.ForwardModel)
print('Population:', tengri.Population)
print('SubModel:', tengri.protocols.SubModel)
"
```

Expected: three class/Protocol names printed without error.

---

## Task 10: Open the PR

- [ ] **Step 1: Push the branch**

The branch was created off `main` via the using-git-worktrees skill. Confirm it tracks origin:

```bash
git status -sb
git push -u origin HEAD
```

- [ ] **Step 2: Open PR**

```bash
gh pr create --title "feat(forward): ForwardModel outer shell (tracer-bullet, single-population)" --body "$(cat <<'EOF'
## Summary

Implements the tracer-bullet vertical slice of the forward-model architecture
spec (`docs/dev/archive/forward-model-architecture.md`):

- `tengri.ForwardModel` — outer shell with `.build(sed=..., observation=...)`
  convenience and `.predict(params)` method.
- `tengri.Population` — frozen dataclass holding one (SED, spatial) pair.
  Spatial is reserved (`None`) in this slice.
- `tengri.protocols.SubModel` — runtime-checkable Protocol with 2 methods
  (`run`, `declared_parameters`).
- `_LegacySEDSubModel` — explicit migration shim that wraps the existing
  `SEDModel` as a `SubModel`. Underscore-prefixed; deleted when the next plan
  factors `SEDModel`'s chain into a first-class `SubModel`.

The new shell is purely additive: no existing class is renamed, deleted, or
restructured. `forward.predict(params)` produces the same numerical result
as `sed_model.predict_photometry(params)` for the common case; the
regression test `test_predict_matches_legacy_sedmodel` enforces this.

End-to-end `@jax.jit` works transparently through the shell
(`test_forward_model_end_to_end_jit`).

## Out of scope (subsequent plans)

- `SpatialComponent` Protocol, `SpatialModel`, `SpatialModelComponent`.
- `SpatialSEDModel` joint mode.
- Multi-population (>1 population per `ForwardModel`) — currently raises
  `NotImplementedError` with an explicit reference to the ADR-0012 plan.
- Parameter namespacing `<population>.<prefix>_<param>` (ADR-0012).
- Refactoring `SEDModel`'s outer shell out of the monolithic class.

## Test plan

- [ ] `tests/unit/protocols/test_submodel_protocol.py` — SubModel Protocol shape.
- [ ] `tests/unit/forward/test_population.py` — Population dataclass.
- [ ] `tests/unit/forward/test_sed_submodel_adapter.py` — legacy adapter contract.
- [ ] `tests/unit/forward/test_forward_model.py` — build + predict.
- [ ] `tests/integration/test_forward_model_jit.py` — end-to-end JIT + closure audit.
- [ ] `pytest tests/ -q` — full regression suite passes (no behavior change for the existing forward path).
EOF
)"
```

- [ ] **Step 3: Report the PR URL to the user**

---

## Self-review checklist

After all tasks are complete, the implementer verifies before declaring done:

- [ ] Each new file has a module docstring that names the architecture-spec section it implements.
- [ ] `SubModel` is exported from `tengri.protocols` (`__init__.py`).
- [ ] `ForwardModel` and `Population` are exported from `tengri` top-level.
- [ ] `pytest tests/ -q` — full suite passes (or all failures are pre-existing, confirmed via `git stash && pytest && git stash pop`).
- [ ] `ruff check && ruff format --check` — clean.
- [ ] CHANGELOG entry under `## Unreleased`.
- [ ] PR description names the out-of-scope items so the reviewer doesn't expect spatial / multi-pop.
- [ ] No file deleted, no existing class renamed.
- [ ] `_LegacySEDSubModel` is underscore-prefixed and its docstring states it is a migration shim.
