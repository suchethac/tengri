# AGN Energy Ledger — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the composable AGN runner energy-conserving by default — the disc's `L_bol` is debited by the reprocessed fractions so `disc(1−Σf) + Σ reprocessor(f)` conserves `L_bol`, reproducing the monolithic models and fixing the #916 root cause.

**Architecture:** Add an `agn_norm="conserving"` **opt-in** policy to `compose_l_nu` in `blocks/runner.py`. Under it, the runner debits the disc `L_λ` by `(1 − agn_torus_frac)` before summing (skipping self-contained tori `none`/`qsogen`/`grahsp`). Torus blocks already normalize to `agn_torus_frac·L_bol`, so only the disc side changes. The change is a single branchless multiply gated on the static policy string + torus name (JIT-safe). **The default stays `cigale_joint` in Phase 1** — flipping it lands in Phase 2 with the CIGALE-reproduction update + recipes (the reproduction relies on the cigale_joint default).

**Tech Stack:** JAX (pure functions, `jnp`, JIT), pytest, chex.

## Global Constraints

- Pure JAX, JIT-compatible: policy dispatch on static Python strings at trace time; never trace `agn_norm` or the registry.
- Units: `L_λ` [erg/s/Å] inside the runner; `L_ν = L_λ·λ²/c` at output.
- Physical constants from `utils/physics_constants` (via `blocks` imports); no local literals.
- American English in all prose/comments.
- Run tests with `.venv/bin/python -m pytest` from the canonical repo dir; `JAX_PLATFORMS=cpu`.
- No back-compat: the ledger is the behavior; existing composable configs move to the conserved numbers.

---

### Task 1: Conservation-invariant test (the test that would have caught #916)

**Files:**
- Test: `tests/physics/agn/test_energy_conservation.py` (create)

**Interfaces:**
- Consumes: `tengri.components.agn.blocks.registry.composable(wave, agn_log_lbol, *, agn_disc_block, agn_torus_block, agn_norm, **params) -> L_ν`
- Produces: the `conservation` marker contract for later tasks.

- [ ] **Step 1: Write the failing test**

```python
# tests/physics/agn/test_energy_conservation.py
"""Energy conservation of the composable AGN runner (conserving policy)."""

import numpy as np
import pytest

from tengri.components.agn.blocks.registry import composable

pytestmark = pytest.mark.conservation

_WAVE = np.geomspace(1e2, 1e7, 2000)   # Å, wide enough to capture disc UV + torus IR


def _band_energy(l_nu, wave):
    nu = 2.99792458e18 / wave           # Å/s
    order = np.argsort(nu)
    return np.trapezoid(np.asarray(l_nu)[order], nu[order])


@pytest.mark.parametrize("torus", ["silva04", "cat3d_wind", "simple", "two_temperature"])
def test_conserving_policy_is_invariant_under_torus_frac(torus):
    """Under agn_norm='conserving', total emitted energy must not change as
    agn_torus_frac moves energy from disc to torus (torus reprocesses disc
    light; bolometric is conserved). This is the invariant #916 violated."""
    e0 = _band_energy(
        composable(_WAVE, 45.0, agn_disc_block="powerlaw", agn_torus_block=torus,
                   agn_norm="conserving", agn_lum_ratio=1.0, agn_torus_frac=0.0), _WAVE)
    for tf in (0.3, 0.6, 0.9):
        e = _band_energy(
            composable(_WAVE, 45.0, agn_disc_block="powerlaw", agn_torus_block=torus,
                       agn_norm="conserving", agn_lum_ratio=1.0, agn_torus_frac=tf), _WAVE)
        assert e == pytest.approx(e0, rel=0.02), (
            f"{torus}: energy not conserved at torus_frac={tf} "
            f"({e:.3e} vs {e0:.3e}) — disc not debited")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/physics/agn/test_energy_conservation.py -q`
Expected: FAIL — energy grows with torus_frac (disc not debited), and/or `agn_norm='conserving'` unknown.

- [ ] **Step 3: (implementation lands in Task 2)** — leave test red, proceed to Task 2.

---

### Task 2: Implement the `conserving` policy (disc debit) in the runner

**Files:**
- Modify: `src/tengri/components/agn/blocks/runner.py` (in `compose_l_nu`, the disc-normalization region ~L466–520 and the policy read ~L484)

**Interfaces:**
- Consumes: existing `compose_l_nu` params incl. `agn_torus_frac` (already flowing through `**params`).
- Produces: `agn_norm="conserving"` behavior; the default value of `agn_norm` becomes `"conserving"`.

- [ ] **Step 1: Change the policy default and add the conserving debit**

In `compose_l_nu`, the policy is read as `_agn_norm = params.get("agn_norm", "cigale_joint")`. Change the default and add the disc debit. After `L_lambda_disc = L_lambda_disc * _disc_ext` (the reddening line, ~L520), insert:

```python
    # ── Energy ledger (agn_norm="conserving", the default) ───────────────
    # The disc carries the intrinsic L_bol; reprocessors (torus) debit it.
    # disc_observed = (1 - agn_torus_frac) * disc, so
    # disc(1-f) + torus(f) conserves L_bol — reproducing the monolithic
    # models (e.g. silva04_agn passes agn_lum_ratio=1-agn_torus_frac to the disc).
    # Static Python branch on the policy string (JIT-safe); the torus block
    # already normalizes its output to agn_torus_frac * L_bol.
    if _agn_norm == "conserving" and agn_torus_block not in _SELF_CONTAINED_TORI:
        _cons_torus_frac = jnp.clip(jnp.asarray(params.get("agn_torus_frac", 0.5)), 0.0, 1.0)
        L_lambda_disc = L_lambda_disc * (1.0 - _cons_torus_frac)
```

Self-contained tori (`_SELF_CONTAINED_TORI = {"none", "qsogen", "grahsp"}`,
runner.py:78) bundle disc+torus self-normalized and bypass the debit; `none`
also covers disc-only configs (no reprocessor → no debit).

**Do NOT flip the default in Phase 1.** Keep `_agn_norm = params.get("agn_norm",
"cigale_joint")`. `conserving` is opt-in — the tests and the Phase-4 retirement
presets pin it explicitly. Flipping the default would silently change the CIGALE
§9 reproduction (which relies on the `cigale_joint` default); that flip lands in
Phase 2 alongside the reproduction update + recipes.

- [ ] **Step 2: Run the conservation test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/physics/agn/test_energy_conservation.py -q`
Expected: PASS — energy invariant across `torus_frac` for all four tori (rel ≤ 0.02).

- [ ] **Step 3: Commit**

```bash
git add tests/physics/agn/test_energy_conservation.py src/tengri/components/agn/blocks/runner.py
git commit -m "feat(agn): energy-conserving composable runner (conserving policy, new default)"
```

---

### Task 3: Monolithic-equivalence gate under `conserving` (full-SED, not peak)

**Files:**
- Test: `tests/regression/agn/test_conserving_matches_monolithic.py` (create)

**Interfaces:**
- Consumes: `composable(...)`; the monolithic `tengri.components.agn.unified.silva04_agn` / `adaf_agn` / `cat3d_wind_agn` (still-defined functions).
- Produces: the per-model equivalence contract that green-lights Phase-4 retirement.

- [ ] **Step 1: Write the failing test**

```python
# tests/regression/agn/test_conserving_matches_monolithic.py
"""The conserving composable path reproduces the monolithic energy-conserving
models across the FULL SED (not just the peak)."""

import numpy as np
import pytest

import tengri.components.agn.unified as U
from tengri.components.agn.blocks.registry import composable

pytestmark = pytest.mark.regression_bug

_WAVE = np.geomspace(1e2, 1e7, 3000)

_CASES = [
    ("silva04",    U.silva04_agn,    dict(agn_disc_block="powerlaw", agn_torus_block="silva04")),
    ("cat3d_wind", U.cat3d_wind_agn, dict(agn_disc_block="powerlaw", agn_torus_block="cat3d_wind")),
    ("adaf",       U.adaf_agn,       dict(agn_disc_block="adaf",     agn_torus_block="silva04")),
]


@pytest.mark.parametrize("name,mono_fn,blocks", _CASES)
def test_conserving_reproduces_monolithic_full_sed(name, mono_fn, blocks):
    for tf in (0.3, 0.5, 0.7):
        mono = np.asarray(mono_fn(_WAVE, 45.0, agn_lum_ratio=1.0, agn_torus_frac=tf))
        comp = np.asarray(composable(_WAVE, 45.0, agn_norm="conserving",
                                     agn_lum_ratio=1.0, agn_torus_frac=tf, **blocks))
        # allclose on the whole curve, normalized to the monolithic peak
        scale = np.max(np.abs(mono))
        np.testing.assert_allclose(comp / scale, mono / scale, atol=2e-2, rtol=0,
            err_msg=f"{name}: conserving preset != monolithic at torus_frac={tf}")
```

- [ ] **Step 2: Run to see which models match and which need block-normalization work**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/regression/agn/test_conserving_matches_monolithic.py -q`
Expected: silva04/cat3d PASS (disc debit is the only gap); adaf may need its disc block to accept the debit — if it fails, record the residual and tighten in a follow-up step (do NOT loosen the tolerance to hide a real shape difference).

- [ ] **Step 3: Commit**

```bash
git add tests/regression/agn/test_conserving_matches_monolithic.py
git commit -m "test(agn): full-SED equivalence gate — conserving preset vs monolithic"
```

---

### Task 4: Guard against regressions in the existing AGN suite

**Files:**
- (no new files) — run the existing AGN tests to catch fallout from the default-policy change.

- [ ] **Step 1: Run the AGN + composable test suites**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/components/agn tests/contract/test_param_groups_agn.py -q`
Expected: green, OR a small set of tests that assumed the old `cigale_joint` default — triage each: if it encodes the old non-conserving behavior, update it to the conserved expectation (no back-compat); if it's a genuine regression, fix the runner.

- [ ] **Step 2: Run ruff**

Run: `.venv/bin/ruff check src/tengri/components/agn/blocks/runner.py tests/physics/agn tests/regression/agn && .venv/bin/ruff format --check src/tengri/components/agn/blocks/runner.py`
Expected: clean.

- [ ] **Step 3: Commit any test updates**

```bash
git add -A && git commit -m "test(agn): update AGN suite for conserving default policy"
```

## Self-Review notes

- Spec coverage: this plan implements Section 1 (ledger disc-debit) + the
  `conserving` policy of Section 3 + the conservation-invariant and
  monolithic-equivalence tests of Section 6. Sections 2 (full block
  shape-provider migration), 3 (cigale_joint/l5100/independent/fagn), 4
  (reddening unification), 5/5b (retirement + recipes), and 7 later phases get
  their own plans.
- The disc-debit is the minimal correct core; polar/line fractions extend the
  same `(1 − Σf)` debit in Phase 1.x once polar is a first-class reprocessor.
- Risk: changing the default from `cigale_joint` to `conserving` alters SKIRTOR
  configs that relied on the implicit default — the CIGALE reproduction recipe
  pins `cigale_joint` explicitly, so the §9 parity test must stay green (checked
  in Task 4 / Phase 2).
