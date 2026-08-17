# SEDModel-as-SubModel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** `SEDModel` directly satisfies the `SubModel` Protocol. The `_LegacySEDSubModel` migration shim introduced in the tracer-bullet (PR #159) is deleted. `ForwardModel.build` and `ForwardModel.predict` deal with `SEDModel` instances directly.

**Architecture:** `SEDModel` gains two methods (`run`, `declared_parameters`) that mirror the existing `_LegacySEDSubModel` adapter. The wrapper class disappears. `ForwardModel`'s shapes are unchanged for callers — the change is internal.

**Tech Stack:** Python 3.12, JAX, pytest, ruff. Project conventions in `CLAUDE.md`.

**Depends on:** PR #159 (tracer-bullet). This branch is stacked on it.

**Out of scope (subsequent plans):**
- Observation Protocol migration (`observation.predict(state, params) → dict`) — `ForwardModel.predict` still reaches `SEDModel.predict_photometry` internally for now.
- Multi-population (ADR-0012).
- Spatial side.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/tengri/forward/sed_model.py` | Modify | Add `run` and `declared_parameters` methods. |
| `src/tengri/forward/forward_model.py` | Modify | Use SEDModel directly; drop `_LegacySEDSubModel` import + wrap. |
| `src/tengri/forward/_sed_submodel_adapter.py` | Delete | Migration shim no longer needed. |
| `tests/unit/forward/test_sed_submodel_adapter.py` | Delete | Tests for deleted shim. |
| `tests/unit/forward/test_sedmodel_as_submodel.py` | Create | New tests: SEDModel satisfies SubModel; runs; declares params. |
| `tests/unit/forward/test_forward_model.py` | Modify | Update imports (no `_LegacySEDSubModel`) — keep coverage. |
| `CHANGELOG.md` | Modify | Note shim deletion under Unreleased. |
| `docs/dev/where-things-live.md` | Modify | Remove the SubModel adapter row, keep others. |

---

## Task 1: Add `SEDModel.declared_parameters` and `SEDModel.run`

**Files:**
- Modify: `src/tengri/forward/sed_model.py` — add two methods on the `SEDModel` class.
- Test: `tests/unit/forward/test_sedmodel_as_submodel.py` (new).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/forward/test_sedmodel_as_submodel.py`:

```python
"""SEDModel directly satisfies the SubModel Protocol.

This makes the _LegacySEDSubModel adapter unnecessary; that file is
deleted in this same plan. The architecture spec at
``docs/dev/archive/forward-model-architecture.md`` §4 has `SEDModel` listed
as one of the three sub-models satisfying SubModel.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.protocols import SubModel
from tengri.protocols.component import ForwardState


@pytest.fixture
def sed_model_minimal(synthetic_ssp, simple_observation):
    from tengri import FIXED, SEDModel

    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl", "*": FIXED},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
    )


def test_sedmodel_satisfies_submodel_protocol(sed_model_minimal) -> None:
    assert isinstance(sed_model_minimal, SubModel)


def test_sedmodel_declared_parameters_matches_spec(sed_model_minimal) -> None:
    declared = sed_model_minimal.declared_parameters()
    assert {d.name for d in declared} == set(sed_model_minimal.spec.free_params)


def test_sedmodel_run_returns_forward_state(sed_model_minimal) -> None:
    state = ForwardState(wave=jnp.array([3000.0, 5000.0, 8000.0]))
    params = {name: 0.5 for name in sed_model_minimal.spec.free_params}
    out = sed_model_minimal.run(state, params)
    assert isinstance(out, ForwardState)


def test_sedmodel_name_attribute(sed_model_minimal) -> None:
    # SubModel requires a `name` attribute. "sed" is conventional for the SED chain.
    assert sed_model_minimal.name == "sed"
```

- [ ] **Step 2: Verify failing**

```bash
PYTHONPATH=/tengri/src \
    /tengri/.venv/bin/pytest \
    tests/unit/forward/test_sedmodel_as_submodel.py -v
```

Expected: all 4 tests fail (no `name`, no `run`, no `declared_parameters`).

- [ ] **Step 3: Implement on SEDModel**

In `src/tengri/forward/sed_model.py`, find the `class SEDModel:` declaration (line ~128). Inside the class body, find an appropriate location near the existing `predict_state` method (~line 3668) and add:

```python
    # ---- SubModel Protocol -------------------------------------------------
    #
    # SEDModel satisfies tengri.protocols.SubModel — the contract that
    # ForwardModel's per-population orchestration consumes. See the
    # forward-model architecture spec (docs/dev/archive/forward-model-architecture.md)
    # §4 for the SubModel layer.

    name: str = "sed"

    def declared_parameters(self):
        """Free parameter declarations for this SED chain.

        Returns
        -------
        list of :class:`tengri.protocols.ParamDeclaration`
            One entry per free parameter, lifted from ``self.spec``.

        Notes
        -----
        Satisfies :class:`tengri.protocols.SubModel`. The orchestrator
        in :class:`tengri.ForwardModel` consumes this aggregated view
        when building the union of parameter declarations across
        populations.
        """
        from tengri.protocols.component import ParamDeclaration

        spec = self.spec
        decls: list[ParamDeclaration] = []
        for pname in spec.free_params:
            prior = _prior_for(spec, pname)
            decls.append(
                ParamDeclaration(
                    name=pname,
                    prior=prior,
                    description="",
                    units="",
                )
            )
        return decls

    def run(self, state, params):
        """Run the SED forward chain. Pure JAX.

        SED is the head of the per-population orchestration; in the
        tracer-bullet single-population path, ``state`` is an empty
        :class:`ForwardState` with just the wavelength grid. This method
        delegates to :meth:`predict_state` for the actual physics.

        Parameters
        ----------
        state : ForwardState
            Incoming state (empty for SED as the head of the chain).
        params : Mapping
            Free parameter values.

        Returns
        -------
        ForwardState
            State with SED contributions populated.
        """
        # SED is currently always the head of the chain. If a future
        # plan needs to thread upstream state (e.g. a spatial-first
        # ResolvedSEDModel), expand this method to merge ``state``.
        return self.predict_state(params)
```

The `name` attribute set at class level provides a default; if `SEDModel` is a dataclass with `name` already declared, skip the assignment.

After the class definition, **add the module-level `_prior_for` helper** if it doesn't already exist (it does NOT today — `_LegacySEDSubModel` had its own copy):

```python
def _prior_for(spec, pname):
    """Locate the prior for ``pname`` on the spec.

    Tries multiple attribute names because the spec's prior-lookup API
    has drifted over time. The same logic was used by the now-deleted
    `_LegacySEDSubModel` adapter.
    """
    for attr_name in ("prior_of", "get_prior", "prior_for"):
        getter = getattr(spec, attr_name, None)
        if callable(getter):
            return getter(pname)
    priors = getattr(spec, "priors", None)
    if priors is not None:
        return priors[pname]
    return None
```

> **Implementer note:** Check the actual spec API. The deleted `_LegacySEDSubModel` used `spec._distributions[pname]` — verify whether `prior_of` / `get_prior` / `priors` exist, and prefer the public one. If only `_distributions` works, use that.

- [ ] **Step 4: Verify passing**

```bash
PYTHONPATH=/tengri/src \
    /tengri/.venv/bin/pytest \
    tests/unit/forward/test_sedmodel_as_submodel.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Lint**

```bash
/tengri/.venv/bin/ruff check \
    src/tengri/forward/sed_model.py \
    tests/unit/forward/test_sedmodel_as_submodel.py
/tengri/.venv/bin/ruff format --check \
    src/tengri/forward/sed_model.py \
    tests/unit/forward/test_sedmodel_as_submodel.py
```

- [ ] **Step 6: Commit**

```bash
git add src/tengri/forward/sed_model.py tests/unit/forward/test_sedmodel_as_submodel.py
git commit -m "feat(forward): SEDModel directly satisfies SubModel Protocol"
```

---

## Task 2: ForwardModel no longer wraps in `_LegacySEDSubModel`

**Files:**
- Modify: `src/tengri/forward/forward_model.py`

- [ ] **Step 1: Update `build`**

In `src/tengri/forward/forward_model.py`, the current `build` does:

```python
from tengri.forward._sed_submodel_adapter import _LegacySEDSubModel
...
sub = sed if not isinstance(sed, SEDModel) else _LegacySEDSubModel(sed)
```

Replace with:

```python
sub = sed
```

(SEDModel now satisfies SubModel directly — no wrapping needed.) Remove the `_LegacySEDSubModel` import at the top of the file. Update the docstring to match.

- [ ] **Step 2: Update `predict`**

Current `predict` reaches:

```python
sed_model = getattr(legacy, "sed_model", None)
if sed_model is None:
    raise NotImplementedError(...)
per_pop[pop.name] = {"phot_fnu": sed_model.predict_photometry(params)}
```

Replace with:

```python
from tengri.forward.sed_model import SEDModel

if not isinstance(pop.sed, SEDModel):
    raise NotImplementedError(
        "ForwardModel.predict currently supports only SEDModel-based "
        "populations. Other SubModel implementations need the "
        "observation-Protocol migration plan."
    )
per_pop[pop.name] = {"phot_fnu": pop.sed.predict_photometry(params)}
```

- [ ] **Step 3: Run all existing forward tests**

```bash
PYTHONPATH=/tengri/src \
    /tengri/.venv/bin/pytest \
    tests/unit/forward/ -v
```

Expected: all forward-model tests still pass (`test_forward_model.py` 6 tests, `test_population.py` 4 tests, `test_sedmodel_as_submodel.py` 4 tests; tests that imported `_LegacySEDSubModel` will fail — those get cleaned up in Task 3).

- [ ] **Step 4: Lint**

```bash
/tengri/.venv/bin/ruff check \
    src/tengri/forward/forward_model.py
```

- [ ] **Step 5: Commit**

```bash
git add src/tengri/forward/forward_model.py
git commit -m "refactor(forward): ForwardModel uses SEDModel directly; drop adapter wrap"
```

---

## Task 3: Delete the adapter and its tests

**Files:**
- Delete: `src/tengri/forward/_sed_submodel_adapter.py`
- Delete: `tests/unit/forward/test_sed_submodel_adapter.py`
- Modify: `tests/unit/forward/test_forward_model.py` — remove references to `_LegacySEDSubModel` if any.

- [ ] **Step 1: Check for remaining references**

```bash
grep -rn "_LegacySEDSubModel\|_sed_submodel_adapter" src/ tests/ docs/
```

Expected: only the file itself and the test file. If `tests/unit/forward/test_forward_model.py` imports `_LegacySEDSubModel` (it does, in `test_build_accepts_explicit_populations` and `test_build_rejects_multi_population_in_tracer_bullet`), update those tests to wrap the SEDModel directly:

Find:
```python
from tengri.forward._sed_submodel_adapter import _LegacySEDSubModel
pop = Population(name="only", sed=_LegacySEDSubModel(sed_model_minimal))
```

Replace with:
```python
pop = Population(name="only", sed=sed_model_minimal)
```

Same for the multi-pop test.

- [ ] **Step 2: Delete the adapter and its tests**

```bash
git rm src/tengri/forward/_sed_submodel_adapter.py tests/unit/forward/test_sed_submodel_adapter.py
```

- [ ] **Step 3: Run forward tests**

```bash
PYTHONPATH=/tengri/src \
    /tengri/.venv/bin/pytest \
    tests/unit/forward/ -v
```

Expected: all remaining forward tests pass (14 = 4 population + 6 forward_model + 4 sedmodel_as_submodel).

- [ ] **Step 4: Lint**

```bash
/tengri/.venv/bin/ruff check src/ tests/
```

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "refactor(forward): delete _LegacySEDSubModel adapter — SEDModel is now a SubModel directly"
```

---

## Task 4: Docs and CHANGELOG

**Files:**
- Modify: `docs/dev/where-things-live.md`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Update where-things-live**

In `docs/dev/where-things-live.md`, find the forward-model assembly table. Remove the row referring to `_LegacySEDSubModel` if present. If the existing rows (`ForwardModel`, `Population`, `SubModel`) need updating to reflect that SEDModel itself now satisfies SubModel, do so. Otherwise, no change needed there.

- [ ] **Step 2: Update CHANGELOG**

Under `## [Unreleased]`, add a new `### Internal` entry (or extend an existing one):

```markdown
- **`SEDModel` directly satisfies `tengri.protocols.SubModel`.** The
  `run(state, params)` and `declared_parameters()` methods are now on
  `SEDModel` itself. The transitional `_LegacySEDSubModel` adapter
  introduced in the tracer-bullet has been deleted; `ForwardModel`
  build and predict paths consume `SEDModel` instances directly. No
  user-visible change to public API.
```

- [ ] **Step 3: Commit**

```bash
git add docs/dev/where-things-live.md CHANGELOG.md
git commit -m "docs: SEDModel satisfies SubModel; adapter deleted"
```

---

## Task 5: Full regression check

- [ ] **Step 1: Targeted forward + protocols tests**

```bash
PYTHONPATH=/tengri/src \
    /tengri/.venv/bin/pytest \
    tests/unit/protocols tests/unit/forward tests/integration/test_forward_model_jit.py \
    tests/contract/test_public_api_surface.py tests/contract/test_public_surface.py -q
```

Expected: all pass.

- [ ] **Step 2: Project-wide lint**

```bash
/tengri/.venv/bin/ruff check src/ tests/
```

Expected: clean.

- [ ] **Step 3: Public import smoke test**

```bash
PYTHONPATH=/tengri/src \
    /tengri/.venv/bin/python -c "
from tengri import ForwardModel, Population, SEDModel
from tengri.protocols import SubModel
sed = SEDModel.__name__
assert hasattr(SEDModel, 'run'), 'SEDModel missing run method'
assert hasattr(SEDModel, 'declared_parameters'), 'SEDModel missing declared_parameters method'
print('OK')
"
```

---

## Task 6: Push + open PR

- [ ] **Step 1: Push**

```bash
git push -u origin HEAD
```

- [ ] **Step 2: PR opened with base `feat/forward-model-tracer-bullet`**

```bash
gh pr create --base feat/forward-model-tracer-bullet \
    --title "refactor(forward): SEDModel directly satisfies SubModel; drop adapter shim" \
    --body "$(cat <<'EOF'
## Summary

Stacked on PR #159 (forward-model tracer-bullet).

`SEDModel` directly satisfies the `SubModel` Protocol now. The `_LegacySEDSubModel` migration shim that the tracer-bullet introduced is deleted.

- Added: `SEDModel.run(state, params) → ForwardState`, `SEDModel.declared_parameters() → list[ParamDeclaration]`, and `SEDModel.name = "sed"` attribute.
- Removed: `src/tengri/forward/_sed_submodel_adapter.py` and its tests.
- Updated: `ForwardModel.build` no longer wraps SEDModel in an adapter; `ForwardModel.predict` calls `pop.sed.predict_photometry(params)` directly when `pop.sed` is a `SEDModel`.

## Public API

Unchanged. `SEDModel`'s existing methods (`predict_photometry`, `predict_spectrum`, `predict_sfh`, etc.) are untouched. Only additive changes (`run`, `declared_parameters`, `name`).

## Out of scope

- Observation Protocol migration (`observation.predict(state, params) → dict`) — `ForwardModel.predict` still uses `predict_photometry`. Subsequent plan.
- Multi-population (ADR-0012).
- Spatial.

## Test plan

- [ ] `tests/unit/forward/test_sedmodel_as_submodel.py` — 4 new tests for SubModel conformance.
- [ ] `tests/unit/forward/test_forward_model.py` — updated to construct `Population` with bare SEDModel; all 6 still pass.
- [ ] `tests/unit/forward/` regression: 14 tests pass total (no shim test file).
- [ ] Contract tests still green.
EOF
)"
```

- [ ] **Step 3: Report the PR URL**

---

## Self-review checklist

- [ ] `_LegacySEDSubModel` is gone (no file, no references).
- [ ] `SEDModel.run`, `SEDModel.declared_parameters`, `SEDModel.name` exist.
- [ ] `isinstance(SEDModel(...), SubModel)` is `True`.
- [ ] All tests in `tests/unit/forward/` pass.
- [ ] No file is renamed.
- [ ] PR base is `feat/forward-model-tracer-bullet` (stacked).
