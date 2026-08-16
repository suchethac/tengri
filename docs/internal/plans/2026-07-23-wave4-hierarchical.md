# Wave 4 Hierarchical Implementation Plan (epic #1322 — #1319 + #1303; Track A of the two-track W4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hierarchical population fits become a `ForwardModel` construction — `mode="hierarchical"` + `shared=(names,)`, data at `fit()` — dissolving `PopulationSEDModel`/`PopulationFitter` into deprecation shims, under the O(1)-in-N scaling contract.

**Architecture:** Rehoming, not rewriting. The working machinery (`PopulationSEDModel.run` vmap, `hierarchical.py` signal_response with `lax.map(batch_size=K)`, `PopulationPosterior`) survives internally; this wave changes who constructs it and where data enters. Four tasks: T1 build-side (`shared=` stored, mode activated), T2 fit-side (data at fit → joint spec materialization → existing engine), T3 the #1303 noise-broadcast bug (fixed in the new path), T4 shims + scaling-contract test.

**Depends on:** Waves 1–2 merged (`mode=` exists reserved; `Data`/ingestion exist; `catalog_z_range` validated helpers exist).

**Timing/scope:** Two-track decision (2026-08-05; spec decisions log 23). This plan is **Track A — standardization, a must**: the `mode="hierarchical"`/`shared=` construction and the Population-class dissolution. Track B — the realistic-N production path — is the *two-step* estimator (MAP/Laplace-fit each galaxy, then fit the population over the interim posteriors); its first instance is merged (PR #1479, shared SFH-PSD σ/τ) and its status, ceiling, and traps live in `docs/dev/hierarchical-psd-handoff.md` and `docs/internal/plans/2026-07-29-hierarchical-psd-recovery.md` — NOT in this plan. Sequencing unchanged: do not start T1–T4 without the orchestrator's explicit go.

## Global Constraints

Identical to `docs/internal/plans/2026-07-23-wave0-api-fixes.md` §Global Constraints, plus:

- **Scaling contract is binding (spec §6.4):** compile O(1) in N; per-galaxy kernels compiled once; `lax.map(batch_size=K)`, never an unrolled N-galaxy graph; no O(N) Python in the hot path. T4's test enforces it — any task breaking it is rejected regardless of green tests.
- **`shared=` is literal one-value sharing** (prior = template's prior). Anything resembling partial pooling (`pooled=`) is out of scope — reject the temptation in review.
- **HMC/MAP focus:** the taught hierarchical method in tests/docstrings is `"vi"`-agnostic — smoke tests use `method="map"` warm-starts and the existing hierarchical VI path only where the legacy engine requires it; no new VI work.

---

### Task T1: build-side — `shared=` on `ForwardModel.build`, hierarchical mode activates

**Executor:** sonnet
**Branch:** `feat/hierarchical-build`

**Files:**
- Modify: `src/tengri/forward/forward_model.py` (`build`: new `shared` kwarg; the Wave-1 `mode=` inference table; a `HierarchicalSpec` carrier on the instance)
- Test: `tests/unit/forward/test_hierarchical_build.py` (create)

**Interfaces:**
- Produces: `ForwardModel.build(sed=template, observation=obs, shared=("sfh_field_psd_sigma", …))` → `fwd.mode == "hierarchical"`, `fwd.shared == tuple(names)`; validation: every shared name ∈ the template's free-parameter names, else ONE error listing the near-misses. `fwd.fit` signature unchanged here (T2 wires it).

- [ ] **Step 1: Failing tests:**

```python
"""#1319: shared= declares the joint structure; mode infers/asserts;
bad names fail with the template's actual free params named."""
import pytest


def test_shared_infers_hierarchical(template_fwd_parts):
    from tengri import ForwardModel
    sed, obs = template_fwd_parts
    fwd = ForwardModel.build(sed=sed, observation=obs,
                             shared=("sfh_field_psd_sigma",))
    assert fwd.mode == "hierarchical"
    assert fwd.shared == ("sfh_field_psd_sigma",)


def test_shared_requires_free_param_of_template(template_fwd_parts):
    from tengri import ForwardModel
    sed, obs = template_fwd_parts
    with pytest.raises(ValueError, match="sfh_field_psd_sigma"):
        ForwardModel.build(sed=sed, observation=obs,
                           shared=("sfh_field_psd_sgima",))   # typo must list candidates


def test_mode_hierarchical_without_shared_errors(template_fwd_parts):
    from tengri import ForwardModel
    sed, obs = template_fwd_parts
    with pytest.raises(ValueError, match="shared="):
        ForwardModel.build(mode="hierarchical", sed=sed, observation=obs)


def test_shared_on_single_mode_errors(template_fwd_parts):
    from tengri import ForwardModel
    sed, obs = template_fwd_parts
    with pytest.raises(ValueError, match="hierarchical"):
        ForwardModel.build(mode="single", sed=sed, observation=obs,
                           shared=("sfh_field_psd_sigma",))
```

`template_fwd_parts`: synthetic-SSP stochastic-SFH template (`sfh={"type": "field", …}` — locate the minimal stochastic build in existing hierarchical tests: `rg -n "PopulationSEDModel|sfh_field_psd" tests/ -l | head`) + a small photometric observation.

- [ ] **Step 2: RED (paste). Step 3: implement.** `shared: tuple[str, ...] | None = None` on `build`; extend the Wave-1 mode-inference table (`shared is not None` ⇒ hierarchical); replace the Wave-1 `NotImplementedError` for asserted-hierarchical-without-kwargs with the precise `ValueError("mode='hierarchical' requires shared=(...) naming the population-level parameters")`. Validation against `sed.spec.free_params` with difflib near-miss suggestions (the error-advice-that-raises rule: every suggested name must be real). Store both on the dataclass (`shared: tuple = ()`).
- [ ] **Step 4: GREEN (paste); sweep `-k "build or mode"`.** Lint. Commit `feat(forward): shared= declares hierarchical structure on ForwardModel.build. Refs #1319`.

---

### Task T2: fit-side — data at `fit()`, joint spec materialized, existing engine reused

**Executor:** sonnet — orchestrator pairs on review
**Branch:** `feat/hierarchical-fit`
**Depends on:** T1.

**Files:**
- Modify: `src/tengri/forward/forward_model.py` (`fit`: hierarchical branch), `src/tengri/forward/population_sed_model.py` (internal constructor accepts pre-validated arrays)
- Test: `tests/inference/test_hierarchical_fit_path.py` (create; mark `slow` where a real fit runs)

**Interfaces:**
- Consumes: T1's `fwd.shared`; Wave-2 `ingest_catalog` (table → `CatalogArrays`); the existing `PopulationSEDModel(sed=template, galaxies=…, shared=…)` + `Fitter` hierarchical routing (`_maybe_extract_batched_data`).
- Produces: `fwd.fit(pop_data, method=…, key=…)` where `pop_data` is a table (ingested like Catalog) or a list of `Data` records — N inferred here; returns the population posterior with `shared_samples`.

- [ ] **Step 1: Map the legacy route precisely.** Paste: `rg -n "_maybe_extract_batched_data|PopulationSEDModel(" src/tengri/inference/fitter.py src/tengri/forward/ | head -10` and the `ForwardModel.build(population=…)` branch. The new path constructs the SAME objects internally: `pop = PopulationSEDModel(sed=self._template, galaxies=<from ingested arrays>, shared=self.shared)` then routes through the existing `Fitter(ForwardModel.build(population=pop, observation=self.observation))` machinery. No inference code changes.
- [ ] **Step 2: Failing tests:**

```python
"""#1319: data enters at fit; N inferred there; the joint fit exposes
shared_samples; the legacy PopulationSEDModel path and the new path
produce the SAME posterior for the same key."""
import jax
import pytest


def test_fit_takes_table_and_infers_n(hier_fwd, pop_table_4gal):
    post = hier_fwd.fit(pop_table_4gal, method="map",
                        key=jax.random.PRNGKey(0))
    assert post.n_galaxies == 4
    assert "sfh_field_psd_sigma" in post.shared_samples


def test_single_mode_rejects_population_table(single_fwd, pop_table_4gal):
    with pytest.raises(TypeError, match="hierarchical"):
        single_fwd.fit(pop_table_4gal, method="map", key=jax.random.PRNGKey(0))


@pytest.mark.slow
def test_new_path_equals_legacy_path(hier_fwd, pop_table_4gal, legacy_population_fit):
    import numpy as np
    post_new = hier_fwd.fit(pop_table_4gal, method="map",
                            key=jax.random.PRNGKey(7))
    post_old = legacy_population_fit(key=jax.random.PRNGKey(7))   # same galaxies, same template, PopulationSEDModel route
    np.testing.assert_allclose(
        np.asarray(post_new.shared_samples["sfh_field_psd_sigma"]),
        np.asarray(post_old.shared_samples["sfh_field_psd_sigma"]), rtol=1e-6)
```

The equivalence test is the load-bearing one: same key ⇒ same numbers proves rehoming-not-rewriting. `legacy_population_fit` builds the current documented route verbatim (template → `PopulationSEDModel` → `ForwardModel.build(population=…)` → `Fitter(...).run`).

- [ ] **Step 3: RED (paste). Step 4: implement** the `fit` hierarchical branch: `if self.mode == "hierarchical":` ingest (`ingest_catalog` when table-like; list-of-`Data` → validate each against the observation, stack — homogeneous-grid contract enforced by the existing `_validate_homogeneous_galaxies`), build the internal `PopulationSEDModel`, delegate. Reject bare `fit(flux, noise)` on hierarchical mode (`TypeError` naming the expected table/records). Reject `fit(pop_data)` on single mode symmetrically.
- [ ] **Step 5: GREEN incl. the slow equivalence test (paste). Step 6:** docstring (Tier 1) on the hierarchical branch of `fit`; lint; commit `feat(forward): hierarchical fit — data at fit(), N inferred, legacy engine reused bit-for-bit. Refs #1319`.

---

### Task T3: the #1303 noise-broadcast crash, fixed on the new path

**Executor:** sonnet
**Branch:** `fix/population-noise-broadcast`
**Depends on:** T2.

**Files:**
- Diagnose in: the hierarchical likelihood assembly (`rg -n "noise_frac_cal|noise_" src/tengri/inference/hierarchical.py src/tengri/inference/loss_functions.py | head`)
- Test: `tests/regression/test_issue_1303_population_noise.py` (create)

- [ ] **Step 1: Reproduce via the NEW path:** hierarchical fit (T2 fixture) with a free noise parameter (`NoiseModel(calibration_floor=Uniform(0.01, 0.15))` on the observation). Expected today: broadcast error — `(N_gal,)` noise param against `(N_gal, n_data)` data. Paste the traceback; identify whether the parameter should be shared (one calibration floor for the survey — physically right for an instrument-level systematic) or per-galaxy.
- [ ] **Step 2: Failing regression test** asserting the fit RUNS and the noise parameter appears where the Step-1 physics decision puts it (shared → in `shared_samples`; per-galaxy → `(N,)` in properties). Encode the decision in the test docstring with one sentence of physical justification.
- [ ] **Step 3: Fix** at the parameter-axis declaration point (the vmap `in_axes`/parameter-axes dict from `population_sed_model.parameter_axes` — noise params need an explicit axis entry, not the default). **Step 4: GREEN (paste);** sweep `-k "population or hierarchical"`; lint; commit `fix(inference): population fits with free noise parameters broadcast correctly. Refs #1303`.

---

### Task T4: deprecation shims + the scaling-contract test

**Executor:** haiku (shims) + sonnet (scaling test)
**Branch:** `feat/hierarchical-shims-scaling`
**Depends on:** T2.

**Files:**
- Modify: `src/tengri/inference/hierarchical.py` (`PopulationFitter` warning text), `src/tengri/forward/population_sed_model.py` (constructor warning), `src/tengri/forward/forward_model.py` (`build(population=…)` warning)
- Test: `tests/regression/test_issue_1319_scaling_contract.py` (create)

- [ ] **Step 1: Shims.** All three legacy entries keep working; each emits the one-shot `DeprecationWarning` pointing at the new construction (`ForwardModel.build(sed=template, shared=(...)); fwd.fit(pop_data, ...)`). The existing `PopulationFitter` warning (hierarchical.py:427-442) gets its message text updated to the new spelling — the mechanism is already there. Tests: one warning-match test per shim.
- [ ] **Step 2: The scaling-contract test** (the wave's exit gate):

```python
"""#1319 / spec 6.4: compile cost O(1) in N. Fitting 2 vs 8 galaxies
(same template, same shapes, same K) must not add compilation work:
we assert via jax's compilation counter, not wall-clock."""
import jax
import pytest

pytestmark = [pytest.mark.regression_bug, pytest.mark.slow]


def test_compile_count_flat_in_n(make_hier_fwd, make_pop_table):
    from jax._src import monitoring  # if unavailable, fall back to
    # jax.monitoring or a compile-log counter: run with
    # jax.log_compiles() capturing logging records and count them.
    counts = {}
    for n in (2, 8):
        fwd = make_hier_fwd()
        table = make_pop_table(n_galaxies=n)
        with _count_compiles() as c:        # implement via jax.log_compiles + logging capture; ~15 lines, include in the test file
            fwd.fit(table, method="map", key=jax.random.PRNGKey(0),
                    forward_chunk_size=2)
        counts[n] = c.count
    assert counts[8] <= counts[2], (
        f"compilations grew with N: {counts} — the joint graph is being "
        "unrolled instead of lax.map'd (scaling contract, spec 6.4)")
```

Implement `_count_compiles` in the test file with `jax.log_compiles()` + a logging handler counting records — complete code, no external tooling. Fresh `make_hier_fwd()` per N isolates model-keyed caches; identical shapes + `forward_chunk_size=2` for both N make the counts comparable.

- [ ] **Step 3: RED-check the gate itself** (neuter protocol): temporarily force the engine to vmap over the full N (set `batch_size=n` … or simulate by asserting on a deliberately-unrolled toy) to confirm the counter DOES detect growth; revert. Paste both runs. **Step 4:** lint; commit `feat(inference): population entry points deprecate toward ForwardModel(shared=...); O(1)-in-N compile gate. Refs #1319`.

---

## Review protocol additions (orchestrator)

T2's same-key equivalence and T4's scaling gate are the two tests I re-run personally. Physics review on T3's shared-vs-per-galaxy noise decision. After the wave: `PopulationSEDModel`/`PopulationFitter` removal is NOT this wave — file the v1.0-removal issue instead.

## Self-review record

- Coverage: #1319 (T1+T2+T4), #1303 (T3), scaling contract (T4), shims (T4). `pooled=` correctly absent (reserved). #1189 sharding correctly absent (post-wave).
- Placeholder scan: `_count_compiles` is specified with its construction; the two "adapt accessor" points carry locator commands; no TBDs.
- Consistency: `shared=` tuple semantics match spec §6.4 and Wave-1 T4's reserved error; `fit(pop_data)` signature matches spec §6.4's example verbatim.
