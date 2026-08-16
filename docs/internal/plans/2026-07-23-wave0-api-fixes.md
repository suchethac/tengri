# Wave 0 API Fixes Implementation Plan (epic #1322)

> **OUTCOME (2026-07-23):** executed; shipped as PRs #1323–#1328. Three tasks proved STALE against origin/main (#1310, #1279, #1306 — closed with evidence), #1177 unverifiable on the pinned blackjax 1.3, #1316 shipped at half scope (blocked on new bug #1329). Both reviewer gates (RED-side neuter on main, diff review) re-run independently. Ledger: the executing job's `progress.md`; verdicts on epic #1322.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the nine Wave-0 correctness fixes plus #1315 from epic #1322 — every one a compile-reuse-contract or fail-loud violation on paths the later API waves rebuild.

**Architecture:** Ten independent tasks, each in its own worktree, each a red→green→regression-test cycle. No task depends on another; any subset can merge. Orchestrator reviews every task against the review checklist before merge (cheap-agent fixes can be silent no-ops — every task therefore *must* paste its RED output).

**Tech Stack:** Python 3.12, JAX (CPU for tests), pytest + chex, ruff. Repo: `suchethac/tengri`.

## Global Constraints

Every task implicitly includes all of these. Violations are review-rejections.

- **Worktree per task.** `git worktree add .claude/worktrees/<branch> -b <branch> origin/main` from the main checkout; work only inside it.
- **Worktree pytest gotcha (MANDATORY):** plain `pytest` in a worktree imports the MAIN checkout. Always run `PYTHONPATH=$PWD/src .venv/bin/pytest …`.
- **Parallelism:** `-n 2` (never `-n auto` — OOMs the machine). Serial debugging: `-n 0`.
- **Lint before every commit:** `.venv/bin/ruff check src/ tests/ && .venv/bin/ruff format src/ tests/`.
- **JAX on CPU:** prefix physics-touching runs with `JAX_PLATFORMS=cpu`.
- **Test taxonomy:** new regression tests go in `tests/regression/`, marked `pytestmark = pytest.mark.regression_bug`, docstring citing the issue number. CI enforces markers (`python tools/check_test_markers.py`).
- **Naming:** American English; canonical names per `docs/dev/NAMING_CONTRACT.md`; units in brackets in every new docstring param.
- **Fix implementation, not tests.** Existing test assertions may not be weakened. If a test seems wrong, STOP and report.
- **Anti-silent-no-op protocol:** every task report MUST contain (a) the pasted FAILING output of the RED step, (b) the pasted PASSING output after the fix, (c) `git diff --stat`. A report missing (a) is rejected unread.
- **Anchor drift:** if a file:line anchor in your task doesn't match what you find, do NOT improvise — run the locator command given in the task, paste what you found, and adapt only if the match is unambiguous; otherwise STOP and report.
- **Scope:** touch only the files your task lists (plus its test file). No drive-by refactors, no comment sprees.
- **Commits:** conventional format (`fix(igm): …`), reference the issue (`Refs #1310`), commit at every green step.
- **Do not push or open PRs** — the orchestrator bundles and merges (review protocol below).

## Orchestrator dispatch & review protocol

**Dispatch:** ≤ 4 tasks concurrently. Executor tier per task header: `haiku` (mechanical), `sonnet` (diagnosis needed). Each dispatch = the task text verbatim + Global Constraints, nothing else.

**Review gate (orchestrator, per task):**
1. RED output present and genuinely failing for the claimed reason.
2. Re-run the acceptance command myself in the task's worktree.
3. **Neuter check** every new guard/regression test: temporarily revert the fix (`git stash push -u -m wave0-neuter-<task>`; restore with `git stash apply <sha>`, then drop) — the new test MUST fail on the unfixed code.
4. Physics review for T1/T2 (equations, IGM conventions) — no cheap-agent doc/physics claim ships unreviewed.
5. `ruff check` clean; markers valid; no out-of-scope diff hunks.

**Merge:** bundle into PRs: PR-A = T4+T5+T6 (registry integrity), PR-B = T7+T8 (test health, label `run-slow-tests`), PR-C = T1, PR-D = T2, PR-E = T3, PR-F = T9, PR-G = T10. Each PR body lists `Fixes #<n>` lines and cites this plan.

---

### Task T1: IGM must not be silently dropped by the WavePrecomp path (#1310)

**Executor:** sonnet
**Branch:** `fix/igm-lut-fail-loud`

**Files:**
- Modify: `src/tengri/forward/sed_model.py` (the `approx=` resolution block, near the `_approx["wave_precomp"]` assignments around lines 832–1060)
- Test: `tests/regression/test_issue_1310_igm_lut.py` (create)
- Read-only context: `src/tengri/observation/observation.py:879-1035` (`predict_via_precomp`; sub-band IGM tensor comment near line ~1023), `src/tengri/observation/photometry.py:553-624` (exact path applies `state.derived["igm_transmission"]`)

**Interfaces:**
- Produces: build-time `ValueError` when a model contains an IGM variant that the WavePrecomp path cannot fold in. Error text MUST name both the IGM type and the two fixes (`approx=None`, or a precomputable mean-IGM type).

**Background:** The exact photometry path applies IGM as an observed-frame transmission. The LUT path only carries IGM inside the sub-band precompute tensor (`stellar_phot_lnu_per_age_subband_igm_precomp`), which exists *"only when a mean-IGM model is precomputable"* (#1135). Any non-precomputable IGM variant + `approx=WavePrecomp()` currently produces fluxes with **no IGM at all**, silently. Contract: fail loud at build.

- [ ] **Step 1: Enumerate IGM variants and the precomputability predicate.** Run and paste:

```bash
rg -n "def igm_absorption|igm_model|_IGM_MODELS|patchy|inoue|madau" src/tengri/components/igm/ | head -30
rg -n "per_age_subband_igm|igm_precomputable|mean.?igm" src/tengri/ | head -20
```

Identify (a) the registry/dispatch listing all IGM `type` strings, (b) the existing condition that decides whether the sub-band IGM tensor is built. If (b) does not exist as a reusable predicate, you will extract one in Step 4.

- [ ] **Step 2: Write the failing parity + guard test.** The test discovers variants programmatically — never hardcode a variant list.

```python
# tests/regression/test_issue_1310_igm_lut.py
"""#1310: IGM must apply on the WavePrecomp path or the build must refuse.

The exact path applies state.derived["igm_transmission"]; the LUT path
folds IGM into the sub-band tensor only for precomputable mean-IGM
models. A non-precomputable variant under approx=WavePrecomp() must
raise at build — never silently drop the IGM.
"""
import pytest

pytestmark = pytest.mark.regression_bug


def _igm_variants():
    # Adapt the import to what Step 1 found; parametrize over ALL
    # registered IGM type strings, splitting precomputable vs not
    # using the same predicate production code uses.
    from tengri.components.igm import igm as igm_mod
    return sorted(igm_mod.available_igm_types())  # adapt name from Step 1


@pytest.mark.parametrize("igm_type", _igm_variants())
def test_lut_build_is_igm_safe(igm_type, synthetic_ssp, minimal_obs):
    """Either the LUT path reproduces the exact path (rtol 1e-3 at z=3
    in the bluest band), or the build raises naming the incompatibility."""
    from tengri import SEDModel, WavePrecomp, Fixed

    def build(approx):
        return SEDModel.build(
            ssp_data=synthetic_ssp, observation=minimal_obs,
            sfh={"type": "dpl"}, igm={"type": igm_type},
            redshift=Fixed(3.0), approx=approx,
        )

    exact = build(None)
    try:
        fast = build(WavePrecomp())
    except ValueError as e:
        assert igm_type in str(e) and "approx" in str(e)
        return  # fail-loud is an accepted outcome
    p = exact.spec.sample_defaults()  # adapt: any valid default param dict
    import numpy as np
    np.testing.assert_allclose(
        fast.predict_photometry(p), exact.predict_photometry(p), rtol=1e-3)
```

Locate the synthetic-SSP and minimal-observation fixtures with `rg -n "synthetic_ssp|def minimal_obs|make_test_ssp" tests/ | head` and adapt the fixture names; if none exists, copy the construction from the nearest existing regression test. If `sample_defaults` doesn't exist, use the pattern the nearest `predict_photometry` test uses to build a param dict.

- [ ] **Step 3: Run to verify it fails.** `JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src .venv/bin/pytest tests/regression/test_issue_1310_igm_lut.py -v -n 0`
Expected: the precomputable variants PASS the parity branch; at least one non-precomputable variant FAILS (parity mismatch, no exception raised). Paste the output.

- [ ] **Step 4: Implement the build-time guard** in the `approx` resolution block of `sed_model.py`: when `self._approx["wave_precomp"]` is being enabled and the model's IGM type fails the precomputability predicate, raise:

```python
raise ValueError(
    f"approx=WavePrecomp() cannot represent igm type {igm_type!r}: "
    "its transmission is not precomputable into the photometric LUT, "
    "so fluxes would silently omit IGM absorption (#1310). Either use "
    "the exact path (approx=None) or a mean-IGM type "
    f"({', '.join(precomputable_types)})."
)
```

Extract the predicate into one function (e.g. `_igm_precomputable(igm_type) -> bool`) used by BOTH this guard and the existing sub-band-tensor decision, so they cannot drift.

- [ ] **Step 5: Re-run the test — all variants must now pass (parity or raise).** Paste output.
- [ ] **Step 6: Full targeted sweep:** `JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src .venv/bin/pytest tests/ -q -n 2 -k "igm or precomp"` — no new failures.
- [ ] **Step 7: Lint + commit.** `git commit -m "fix(igm): WavePrecomp build refuses non-precomputable IGM instead of silently dropping it. Refs #1310"`

---

### Task T2: free-z ztable accuracy gate (#1134)

**Executor:** sonnet
**Branch:** `fix/ztable-accuracy`

**Files:**
- Modify: `src/tengri/forward/sed_model.py:166-230` (`WavePrecomp` defaults) and, only if the harness demands it, the ztable-construction code it points to
- Test: `tests/regression/test_issue_1134_ztable_accuracy.py` (create)

**Interfaces:**
- Produces: default ztable settings under which free-z LUT photometry agrees with the exact path to `< 1%` in every band across the z grid below; a permanent accuracy-gate test.

**Background:** With default `n_z=100`, ztable interpolation error is −4% (des_i, z=0.3) and **+40%** (GALEX FUV, z=1) — the Lyman break sweeping through a blue band between z-nodes. The Catalog design (epic #1322 Wave 2) leans directly on this mechanism, so the default must be safe *before* Wave 2.

- [ ] **Step 1: Write the accuracy harness as the failing test.**

```python
# tests/regression/test_issue_1134_ztable_accuracy.py
"""#1134: free-z ztable photometry must track the exact path to <1%.

Default n_z=100 gave -4% (des_i, z=0.3) and +40% (GALEX FUV, z=1):
linear z-interpolation across the Lyman-break sweep. This test is the
permanent accuracy gate for the ztable defaults.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

BANDS = ["galex_fuv", "galex_nuv", "sdss_u", "des_i"]
Z_GRID = np.linspace(0.05, 1.5, 25)
RTOL = 0.01


def test_ztable_matches_exact_below_1pct(synthetic_ssp):
    from tengri import SEDModel, WavePrecomp, Uniform
    from tengri.observation import Observation, Photometry

    obs = Observation(photometry=Photometry.from_names(BANDS))
    common = dict(ssp_data=synthetic_ssp, observation=obs,
                  sfh={"type": "dpl"}, redshift=Uniform(0.01, 2.0))
    exact = SEDModel.build(approx=None, **common)
    fast = SEDModel.build(approx=WavePrecomp(), **common)  # library default

    worst = 0.0
    for z in Z_GRID:
        p = exact.spec.sample_defaults() | {"redshift": float(z)}  # adapt as in T1
        fe, ff = (np.asarray(m.predict_photometry(p)) for m in (exact, fast))
        rel = np.max(np.abs(ff - fe) / np.abs(fe))
        worst = max(worst, rel)
    assert worst < RTOL, f"worst ztable error {worst:.1%} exceeds {RTOL:.0%}"
```

- [ ] **Step 2: Run — expect FAIL** with worst error near the reported +40%. Paste the number.
- [ ] **Step 3: Find the cheapest passing default empirically.** Loop `n_z ∈ {200, 400, 800}` by passing `WavePrecomp(n_z=…)` in the harness; record worst error and LUT build wall-time for each (`time.perf_counter()` around the `SEDModel.build`). Paste the table.
- [ ] **Step 4: Decide + implement.** If some `n_z ≤ 800` passes: change the `WavePrecomp` field default (`n_z: int = <value>`) and record the measured build-cost delta in the class docstring (numbers, not adjectives). If NONE passes: STOP — report the table; the fix needs break-aware node placement, which is orchestrator-scoped, not yours.
- [ ] **Step 5: Re-run test with library defaults — PASS.** Paste output.
- [ ] **Step 6: Sweep:** `…pytest tests/ -q -n 2 -k "wave_precomp or ztable"` — no new failures. Lint + commit: `fix(forward): raise WavePrecomp n_z default so free-z ztable error stays <1% across bands. Refs #1134`.

---

### Task T3: fit_batch must not recompile per galaxy (#1316)

**Executor:** sonnet
**Branch:** `fix/fit-batch-recompile`

**Files:**
- Modify: `src/tengri/forward/convenience.py:490-538` (`fit_batch` row loop; the per-row clone at 510–520)
- Test: `tests/regression/test_issue_1316_fit_batch_recompile.py` (create)

**Interfaces:**
- Consumes: `WavePrecomp(catalog_z_range=…)` runtime-z mechanism (`sed_model.py:188-212`); the fit-time `params={"redshift": z}` override shown in the `WavePrecomp` docstring example.
- Produces: `fit_batch(redshift_col=…)` on a `catalog_z_range` model runs the loop with **zero** model clones; without `catalog_z_range` it emits one loud `UserWarning` naming the cliff and the fix.

- [ ] **Step 1: Verify the params-override plumbing exists.** Run and paste: `rg -n "params=" src/tengri/forward/convenience.py src/tengri/forward/sed_model.py | rg -n "fit|run" | head`. Confirm a per-fit redshift override reaches the fitter (the `WavePrecomp` docstring example `model.fit(row.data, params={"redshift": row.z})` is the intended route). If no such kwarg exists in the actual `fit` signature, STOP and report — do not invent plumbing.
- [ ] **Step 2: Failing tests** (behavioral: count clones and demand the warning — do not try to count XLA compiles):

```python
# tests/regression/test_issue_1316_fit_batch_recompile.py
"""#1316: fit_batch(redshift_col=...) cloned a new SEDModel per row ->
new compile signature per row -> full recompile per galaxy, silently.
With catalog_z_range the ztable takes z at runtime: one model, one compile."""
import warnings
import pytest

pytestmark = pytest.mark.regression_bug


def test_no_clone_when_catalog_z_range(monkeypatch, synthetic_ssp, tiny_catalog):
    from tengri import SEDModel, WavePrecomp, Fixed
    from tengri.forward import convenience

    model = SEDModel.build(
        ssp_data=synthetic_ssp, observation=tiny_catalog.obs,
        sfh={"type": "dpl"}, redshift=Fixed(0.0),
        approx=WavePrecomp(catalog_z_range=(0.05, 1.5)))

    clones = []
    orig = type(model).__new__
    def counting_new(cls, *a, **k):
        clones.append(cls)
        return orig(cls)
    monkeypatch.setattr(type(model), "__new__", counting_new)
    convenience.fit_batch(model, tiny_catalog.table,
                          flux_cols=tiny_catalog.flux_cols,
                          err_cols=tiny_catalog.err_cols,
                          redshift_col="z", method="map")
    assert clones == [], f"fit_batch cloned {len(clones)} models; expected 0"


def test_warns_without_catalog_z_range(synthetic_ssp, tiny_catalog):
    from tengri import SEDModel, Fixed
    from tengri.forward import convenience
    model = SEDModel.build(ssp_data=synthetic_ssp, observation=tiny_catalog.obs,
                           sfh={"type": "dpl"}, redshift=Fixed(0.0))
    with pytest.warns(UserWarning, match="catalog_z_range"):
        convenience.fit_batch(model, tiny_catalog.table,
                              flux_cols=tiny_catalog.flux_cols,
                              err_cols=tiny_catalog.err_cols,
                              redshift_col="z", method="map")
```

Build `tiny_catalog` (3 rows, 3 bands, z ∈ {0.1, 0.5, 1.0}) as a local fixture from whatever table type `fit_batch` already accepts — read its docstring; keep `method="map"` (HMC/MAP focus; fastest stable backend).

- [ ] **Step 3: Run — both FAIL** (clones observed; no warning emitted). Paste.
- [ ] **Step 4: Implement** in the row loop: if `redshift_col is not None` and `getattr(model, "_catalog_z_range", None)` is set and covers the catalog's z span → skip the clone entirely, pass the row z via the params override from Step 1. If `_catalog_z_range` is `None` → `warnings.warn("fit_batch(redshift_col=...) without WavePrecomp(catalog_z_range=...) recompiles the forward model for EVERY row (one compile per galaxy). Build the model with approx=WavePrecomp(catalog_z_range=(zmin, zmax)) to compile once. See #1316.", UserWarning, stacklevel=2)` and keep the old path. If the span exceeds `_catalog_z_range` → `ValueError` naming both ranges.
- [ ] **Step 5: Run — PASS.** Paste. **Step 6:** sweep `-k "fit_batch"`, lint, commit `fix(forward): fit_batch reuses one compiled model under catalog_z_range; warns loudly otherwise. Refs #1316`.

---

### Task T4: eight declared parameters missing from the registry (#1307)

**Executor:** haiku
**Branch:** `fix/param-registry-eight`

**Files:**
- Modify: the parameter-description registry (locate: `rg -ln "def describe_parameter" src/tengri/` — expected in `src/tengri/parameters/`)
- Test: `tests/regression/test_issue_1307_registry_params.py` (create)

**The eight, with owning components** (copy units/descriptions FROM the component class declarations — they are the source of truth; do not write physics from memory): `gamma`, `delta_alpha_ox`, `e_cut` (agn_xray_corona) · `T_max` (powerlaw_disc) · `det_hmxb`, `det_lmxb` (xray_aird) · `polar_temperature`, `delta` (skirtor polar).

- [ ] **Step 1: Failing test** — parametrize `tengri.describe_parameter(name)` over the eight **fully-prefixed** names (find each component's `parameter_prefix` in its class; e.g. the corona component's prefix + `gamma`). Assert it returns without raising and the result mentions a unit (every entry must satisfy the units-in-brackets rule; dimensionless is spelled `[dimensionless]` or `units=""` per the registry's own convention — copy the convention from an existing entry).

```python
import pytest
pytestmark = pytest.mark.regression_bug

MISSING = [
    # fill each with the exact prefixed name found in Step 1 discovery,
    # e.g. "xray_corona_gamma" — paste the rg output proving each prefix
]

@pytest.mark.parametrize("name", MISSING)
def test_describe_parameter_knows_component_params(name):
    import tengri
    info = tengri.describe_parameter(name)   # must not raise (was KeyError)
    assert info  # non-empty description
```

- [ ] **Step 2: Run — 8 FAILs (KeyError).** Paste. **Step 3:** add the eight registry entries, copying `units=`/description text verbatim from each component's class-level `Distribution` declarations. **Step 4:** run — PASS. **Step 5:** stronger guard while you're here: if the registry module already has a completeness test (rg `"registry" tests/ -l | rg param`), extend it to iterate ALL registered components' declared parameters; if that surfaces more than these eight, report the extra names — do NOT fix beyond the eight (scope). **Step 6:** lint, commit `fix(api): register the eight component-declared params describe_parameter raised on. Refs #1307`.

---

### Task T5: `dh02_ce01` advertised but unbuildable (#1279)

**Executor:** haiku
**Branch:** `fix/dust-menu-dh02`

**Files:**
- Modify: `src/tengri/parameters/groups.py:316` (declared emission list) and/or `src/tengri/forward/component_factory.py` (`_EMISSION_TYPE_ALIASES`)
- Test: `tests/regression/test_issue_1279_dust_menu.py` (create)

- [ ] **Step 1: Is there an implementation?** Run and paste: `rg -in "dh02|dale.?helou|ce01|chary" src/tengri/components/dust/ src/tengri/forward/component_factory.py`. Decision rule: an existing component that IS Dale & Helou 2002 (+Chary–Elbaz) under another registry key → add `"dh02_ce01"` to `_EMISSION_TYPE_ALIASES` mapping to it. Nothing → remove `'dh02_ce01'` from the `groups.py` declared list (advertising the unbuildable is the bug; salvage first, delete second).
- [ ] **Step 2: Failing guard test** (general, not name-specific — prevents the whole class):

```python
import pytest
pytestmark = pytest.mark.regression_bug

def test_every_advertised_dust_emission_type_builds(synthetic_ssp, minimal_obs):
    """#1279: the build grammar's advertised dust-emission menu must be
    a subset of what the registry can actually construct."""
    from tengri import SEDModel, FIXED
    from tengri.parameters import groups
    for name in groups.valid_dust_emission_types():   # adapt: the menu source found in Step 1
        SEDModel.build(ssp_data=synthetic_ssp, observation=minimal_obs,
                       sfh={"type": "dpl"},
                       dust={"type": "two_component", "all_params": FIXED,
                             "emission": {"type": name, "all_params": FIXED}})
```

- [ ] **Step 3: Run — FAILS on `dh02_ce01`** (`ValueError: … not found in registry`). Paste. **Step 4:** apply the Step-1 decision. **Step 5:** run — PASS over the whole menu. **Step 6:** also assert `"dh02_ce01" in builders.dust.emission.available()` **iff** you aliased (else assert absent). Lint, commit `fix(dust): stop advertising unbuildable dh02_ce01 (menu ⊆ registry guard). Refs #1279`.

---

### Task T6: four registered components unreachable from the grammar (#1120)

**Executor:** haiku
**Branch:** `fix/grammar-reach-xray-radio`

**Files:**
- Modify: `src/tengri/parameters/groups.py` (`_valid_xray_types()` / `_valid_radio_types()`)
- Test: `tests/regression/test_issue_1120_grammar_reach.py` (create)

The four: `xray_aird`, `agn_xray_corona` (xray axis) · `radio_powerlaw`, `radio_dpl` (radio axis). This is the #1291 pattern (radio sub-block invisible to discovery) on two more axes — mirror that merged fix's shape (`git log --oneline --grep 1291`, read the diff).

- [ ] **Step 1: Failing test** — the inverse guard of T5 (registry ⊆ menu, per axis):

```python
import pytest
pytestmark = pytest.mark.regression_bug

@pytest.mark.parametrize("axis,expected", [
    ("xray", {"xray_aird", "agn_xray_corona"}),
    ("radio", {"radio_powerlaw", "radio_dpl"}),
])
def test_registered_components_reachable_from_grammar(axis, expected):
    """#1120: every _REGISTRY component of an axis must appear in that
    axis's grammar menu, or it is silently unreachable."""
    from tengri.parameters import groups
    menu = set(groups.valid_types_for(axis))  # adapt to the actual accessor names
    assert expected <= menu, f"unreachable on {axis}: {expected - menu}"
```

- [ ] **Step 2: Run — FAIL naming the four.** Paste. **Step 3:** add them to the two `_valid_*_types()` menus; if a type needs default params to be buildable, verify with a `SEDModel.build(…, xray={'type': 'xray_aird', 'all_params': FIXED})` smoke inside the test (extend the test to build each newly reachable type, T5-style). **Step 4:** run — PASS; sweep `-k "grammar or discovery or menu"`. **Step 5:** lint, commit `fix(api): xray_aird/agn_xray_corona/radio_powerlaw/radio_dpl reachable from the build grammar. Refs #1120`.

---

### Task T7: slow-tier test hard-codes a broken-tier backend (#1305)

**Executor:** haiku
**Branch:** `fix/vi-smoke-tier-guard`

**Files:**
- Modify: `tests/integration/test_vi_smoke_per_emitter.py:220` (and any sibling call sites in the same file)

- [ ] **Step 1: Reproduce.** `JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src .venv/bin/pytest tests/integration/test_vi_smoke_per_emitter.py -q -n 0 -m slow` — paste the failure (broken-tier method).
- [ ] **Step 2: Implement the self-healing guard** (NOT a bare skip — it must auto-revive when the tier changes):

```python
import pytest
from tengri.inference._registration import get_backend  # adapt import via rg "def get_backend"

def _tier(method):
    return getattr(get_backend(method), "tier", "stable")

pytestmark_native = pytest.mark.skipif(
    _tier("native_vi_linear") == "broken",
    reason="#1305: native_vi_linear registered tier='broken'; "
           "smoke auto-revives when the tier is repaired",
)
```

Apply the marker to exactly the tests that invoke the broken-tier method. Per the epic's HMC/MAP focus, do NOT switch these tests to another VI backend and do NOT attempt backend repair.

- [ ] **Step 3: Re-run — SKIPPED with the #1305 reason** (paste), and the rest of the file green. **Step 4:** neuter-proof: temporarily fake `_tier` to return `"experimental"` and confirm the test RUNS (then revert the fake). Paste both. **Step 5:** lint, commit `fix(tests): tier-aware skip for native_vi_linear smoke; auto-revives on repair. Refs #1305`.

---

### Task T8: eline_mode='fitted' registers no amplitude parameters (#1306)

**Executor:** sonnet
**Branch:** `fix/eline-fitted-amplitudes`

**Files:**
- Diagnose in: `src/tengri/inference/fitter.py` (amplitude merge near line 815; `_eline_fitted` resolution near the `__init__` eline block), `src/tengri/observation/spectroscopy.py` (`eline_mode` handling)
- Test: the four failing tests in `tests/inference/test_eline_fitting.py::TestFittedMode` are the RED state — do not write new tests until Step 4.

- [ ] **Step 1: Reproduce.** `JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src .venv/bin/pytest "tests/inference/test_eline_fitting.py::TestFittedMode" -v -n 0 -m slow` — paste all four failures.
- [ ] **Step 2: Instrument, don't guess.** In a scratch script (under `$CLAUDE_JOB_DIR/tmp`, not the repo), build the exact fixture the failing test builds, then print: `fitter._eline_fitted`, the eline catalog resolved, and `fitter._free_names`. The known suspects, in order: (a) `_eline_fitted` never set truthy because the `Spectroscopy(eline_mode="fitted")` flag is read from the wrong place; (b) amplitudes merged into a params dict but not into `_free_names`, so the sampler never sees them. Paste which it is, with the line number.
- [ ] **Step 3: Fix at the root** (the resolution or the merge — not the tests, not the assertions). The fix must make amplitude parameter names (whatever prefix the catalog uses — check `LineList`/eline naming, prefix rule NAMING_CONTRACT §3.2) appear in `fitter._free_names`.
- [ ] **Step 4: Regression distillation.** The slow tests passing is necessary but they cost minutes; add one FAST unit regression in `tests/regression/test_issue_1306_eline_fitted.py`: construct the Fitter with `eline_mode="fitted"` (no fit call) and assert the amplitude names are in `fitter._free_names`. Marker `regression_bug`, docstring citing #1306 and the root cause you found.
- [ ] **Step 5:** all four slow tests + the new fast test green (paste); sweep `-k eline`; lint; commit `fix(inference): eline_mode='fitted' registers amplitude free parameters (root: <one-line cause>). Refs #1306`.

---

### Task T9: mclmc backends broken under blackjax ≥ 1.5 (#1177)

**Executor:** sonnet
**Branch:** `fix/mclmc-blackjax15`

**Files:**
- Modify: `src/tengri/inference/backends/mcmc/mclmc.py`
- Test: `tests/regression/test_issue_1177_mclmc_blackjax.py` (create)

- [ ] **Step 1: Establish the installed reality.** Paste: `.venv/bin/python -c "import blackjax; print(blackjax.__version__)"` and the current failure: a minimal `run_mclmc` invocation on any existing tiny inference fixture (find one: `rg -ln "mclmc" tests/ | head`; if a test exists, run it; expected `TypeError` on the kernel/adaptation interface).
- [ ] **Step 2: Read the ≥1.5 interface in the installed package** — `.venv/bin/python -c "import blackjax, inspect; print(inspect.signature(blackjax.mclmc)); print(inspect.signature(blackjax.mclmc_find_L_and_step_size))"` (adapt names from what `dir(blackjax)` shows). Do NOT code from memory of old blackjax; code against the pasted signatures.
- [ ] **Step 3: Failing smoke test** — 2-D standard normal logdensity, `run_mclmc` for 100 steps, assert finite samples with sample-mean within 0.5 of zero. Mark `pytestmark = [pytest.mark.regression_bug]`; keep it seconds-fast (no SED model — mclmc runners accept a bare logdensity via the InferenceContext primitives; mirror how the existing mclmc test builds its target, or drive the backend's inner kernel-construction function directly if it is importable).
- [ ] **Step 4: Fix** `mclmc.py` against the installed ≥1.5 API, keeping ≤1.4 compat only if free via `hasattr` feature-detection (no version parsing). If the two adaptation APIs are irreconcilable in one code path, drop pre-1.5 support and declare the floor in the module docstring + `requires=` metadata in `_registration.py` — smaller honest surface beats a fake compat matrix.
- [ ] **Step 5:** smoke PASS (paste); any pre-existing mclmc tests un-skip/pass; lint; commit `fix(inference): mclmc backends work under blackjax>=1.5 kernel/adaptation API. Refs #1177`.

---

### Task T10: ForwardModel.build inherits the observation; LUT-scoped mismatch guard (#1315)

**Executor:** sonnet
**Branch:** `fix/forward-obs-inherit`

**Files:**
- Modify: `src/tengri/forward/forward_model.py:311-402` (`build`; `observation: Any` mandatory at :318)
- Test: `tests/regression/test_issue_1315_obs_inherit.py` (create)

**Interfaces:**
- Consumes: `sed.observation` (may be `None`); the LUT flag `sed._approx["wave_precomp"]`; filter content-hash pattern from `sed_model.py:3313` (`hash(tuple(np.asarray(t).tobytes() for t in filter_trans))`).
- Produces: `ForwardModel.build(sed=…)` with `observation` omitted inherits `sed.observation`; passing a different-filter observation raises **only** when the sed carries a baked wave-precomp LUT. `populations=`/`population=` branches still require `observation` explicitly.

- [ ] **Step 1: Failing tests, four cases:**

```python
import pytest
pytestmark = pytest.mark.regression_bug


def _fingerprint(obs):
    import numpy as np
    return hash(tuple(np.asarray(t).tobytes() for t in obs.photometry.filter_trans))


def test_omitted_observation_inherits(synthetic_ssp, obs_a):
    from tengri import SEDModel, ForwardModel
    sed = SEDModel.build(ssp_data=synthetic_ssp, observation=obs_a, sfh={"type": "dpl"})
    fwd = ForwardModel.build(sed=sed)                      # no observation kwarg
    assert _fingerprint(fwd.observation) == _fingerprint(obs_a)


def test_different_filters_no_lut_allowed(synthetic_ssp, obs_a, obs_b):
    from tengri import SEDModel, ForwardModel
    sed = SEDModel.build(ssp_data=synthetic_ssp, observation=obs_a, sfh={"type": "dpl"})
    fwd = ForwardModel.build(sed=sed, observation=obs_b)   # exact path: fine
    assert _fingerprint(fwd.observation) == _fingerprint(obs_b)


def test_different_filters_with_lut_raises(synthetic_ssp, obs_a, obs_b):
    from tengri import SEDModel, ForwardModel, WavePrecomp
    sed = SEDModel.build(ssp_data=synthetic_ssp, observation=obs_a,
                         sfh={"type": "dpl"}, approx=WavePrecomp())
    with pytest.raises(ValueError, match="LUT"):
        ForwardModel.build(sed=sed, observation=obs_b)


def test_population_branch_still_requires_observation(synthetic_ssp, obs_a):
    from tengri import ForwardModel
    with pytest.raises(TypeError):
        ForwardModel.build(populations=[])                 # no obs: must not silently pass
```

`obs_a`/`obs_b`: two ≥2-band observations with different filter sets (e.g. sdss_g+sdss_r vs sdss_i+sdss_z) via `Photometry.from_names`.

- [ ] **Step 2: Run — FAIL** (today: omitting `observation` is a `TypeError` since it is positional-required). Paste.
- [ ] **Step 3: Implement.** `observation: Any | None = None` in the signature; resolution at the top of `build`:

```python
if observation is None:
    if sed is None:
        raise TypeError(
            "ForwardModel.build(populations=...)/(population=...) requires "
            "observation=... explicitly (no single sed to inherit from).")
    observation = getattr(sed, "observation", None)
    if observation is None:
        raise TypeError(
            "ForwardModel.build needs observation=... (the sed carries none).")
elif sed is not None:
    sed_obs = getattr(sed, "observation", None)
    if (sed_obs is not None
            and _filters_fingerprint(sed_obs) != _filters_fingerprint(observation)
            and getattr(sed, "_approx", {}).get("wave_precomp")):
        raise ValueError(
            "This sed carries a WavePrecomp LUT integrated against different "
            "filters than the observation passed to ForwardModel.build — its "
            "photometry would be silently wrong (#1315). Rebuild the sed with "
            "this observation, or build it without approx=.")
```

with `_filters_fingerprint` as a module-level helper using the `sed_model.py:3313` tobytes-hash pattern (handle `photometry is None` → sentinel). Keep the docstring's Parameters section updated (`observation : object, optional — inherited from ``sed`` when omitted`).

- [ ] **Step 4: Run — 4/4 PASS.** Paste. **Step 5:** sweep `…pytest tests/ -q -n 2 -k "forward_model or build"` — the mandatory-observation change must break nothing (existing callers all pass it explicitly). **Step 6:** lint, commit `fix(forward): ForwardModel.build inherits sed.observation; raises only on baked-LUT filter mismatch. Refs #1315`.

---

## Successor plans (not in this document — do not start)

- **Wave 1b plan** (`#1321` razor: `Data` record, `Observation.lines=`, deprecation shims, `mode=`) — orchestrator writes after Wave 0 review; its file map touches `observation/observation.py`, a new `observation/data.py`, `forward/forward_model.py`.
- **Wave 2 plan** (`#1317` Catalog + `#1318` lean/prewarm + `#1313`) — after Wave 1b lands; defaults `method="map"`/`"mcmc_nuts"` per the HMC/MAP focus.
- **Wave 3/4 plans** (notebooks + `#232` split; hierarchical `#1319`) — post Paper-I scheduling.

## Plan self-review record

- Spec coverage: all 9 Wave-0 epic checkboxes have a task (T1–T9) + T10 = #1315; #1232/#307 closed as stale, correctly absent.
- Placeholder scan: the only "adapt" points are locator-command-driven name adaptations with explicit STOP-on-ambiguity clauses — no TBDs, every code step shows code.
- Type consistency: fixture names (`synthetic_ssp`, `minimal_obs`, `obs_a/obs_b`, `tiny_catalog`) are task-local; each task that uses one carries its own locate-or-build instruction.
