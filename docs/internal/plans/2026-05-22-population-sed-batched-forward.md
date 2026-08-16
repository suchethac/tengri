# PopulationSEDModel batched forward path — implementation plan

> **Status:** Plan only. Last remaining deliverable from issue #211. Do not start implementation until the user reviews this plan.

**Goal:** Move hierarchical-population inference out of `tengri.inference.hierarchical.PopulationFitter` (the legacy machine that the `Fitter` currently delegates to) and into the forward-side pipeline — `PopulationSEDModel.run` runs the SED template under `jax.vmap` across N galaxies and produces a batched `ForwardState`, the observation produces a batched prediction, the likelihood reduces in a single `sum` across the galaxy axis, and ordinary `Fitter` machinery (NUTS, VI, etc.) drives the joint posterior.

**Architecture:** Pure forward-side composition. `Fitter` does not know it is a hierarchical fit; it sees a `ForwardModel`, calls `forward.predict(params)`, gets back `{"phot_fnu": (N_galaxies, n_filters)}`, hands it to the likelihood with batched data. The "shared parameters" concept is enforced by `PopulationSEDModel` choosing which parameters get broadcast (shared) vs which get the per-galaxy `vmap` axis.

**Why this matters:** Today's path delegates from `Fitter` to `PopulationFitter`. That works but it's a special-case branch in the inference layer. After this refactor, the inference layer has zero special-case code for hierarchical fits — it's just a `ForwardModel` with a particular SubModel shape.

## Open design choices (REQUIRES USER REVIEW)

### 1. Per-galaxy parameter naming

Hierarchical fits today have two parameter scopes:
- **Shared:** `sfh_field_psd_sigma`, `sfh_field_psd_tau_myr` — one value across all galaxies
- **Per-galaxy:** `sfh_dpl_alpha`, `sfh_dpl_beta`, … — N values, one per galaxy

The user's params dict needs to encode both. Three options:

**A.** Flat namespace, per-galaxy params get a `[i]` suffix:
```python
params = {
    "sfh_field_psd_sigma": 1.0,           # scalar
    "sfh_field_psd_tau_myr": 50.0,        # scalar
    "sfh_dpl_alpha[0]": 1.8,              # per-galaxy
    "sfh_dpl_alpha[1]": 2.1,
    ...
}
```

**B.** Batched arrays in a flat namespace:
```python
params = {
    "sfh_field_psd_sigma": 1.0,                       # scalar (shared)
    "sfh_dpl_alpha": jnp.array([1.8, 2.1, 1.5, ...]), # batched
}
```

**C.** Two-level namespace mirroring ADR-0012:
```python
params = {
    "sfh_field_psd_sigma": 1.0,
    "g0.sfh_dpl_alpha": 1.8,
    "g1.sfh_dpl_alpha": 2.1,
}
```

**Recommendation: B.** Cleanest JAX/vmap shape and lowest namespace churn. The PSD scalar is automatically broadcast across the vmap axis; the per-galaxy 1-D array vmaps naturally.

### 2. Observation batching

The observation also needs to handle a batched state. Two paths:

**A.** Auto-vmap inside `PopulationSEDModel.run`: the SubModel applies `jax.vmap(template.run, in_axes=...)` to the SED template, producing a batched ForwardState. The observation calls `vmap(self.observation.predict)` itself.

**B.** The forward model exposes a batched-aware observation contract: `Observation.predict` learns to handle a batched state in its leading axis. Existing observations gain auto-batch support via `tree_map` over the inputs.

**Recommendation: A.** PopulationSEDModel owns the batch axis; observations stay single-galaxy. Less surface area to change.

### 3. Likelihood batching

The likelihood already reduces over filter / wavelength axes (`sum(((flux_pred - flux_obs) / noise)**2)`). For batched data:

**A.** Likelihood accepts batched arrays automatically (rely on JAX broadcasting + a sum over the new leading axis).

**B.** New explicit `BatchedLikelihood` adapter that wraps a single-galaxy likelihood + an `(N, n_filters)` data shape.

**Recommendation: A.** As long as data and noise are passed as `(N, n_filters)`, JAX broadcasting handles the rest cleanly.

### 4. Migration path for `PopulationFitter`

Once the forward-path works, what happens to `PopulationFitter`?

**A.** Keep it as a back-compat shim: existing direct-API callers see the same DeprecationWarning shipped in #219, the routing path in `Fitter._maybe_population_delegate` becomes a no-op redirect to the new `Fitter(forward).run` path.

**B.** Delete `PopulationFitter` entirely once #211's last item lands. The `_via_routing=True` flag goes away with it.

**Recommendation: A.** Soft cut. The shim costs one file (~2k lines) but lets existing scripts keep working until the v1.0 cut.

## Out of scope (post-v1)

- Per-galaxy `jax.vmap` across spatial + SED. The `PopulationSpatialSEDModel` *(far future)* composition lands separately.
- Resolved-imaging hierarchical fits.
- Per-galaxy redshift fits where each galaxy has its own redshift prior. This needs to drop into the shared-vs-batched-vs-per-galaxy decision tree and is a small generalization once (1) lands.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/tengri/forward/population_sed_model.py` | Modify | `PopulationSEDModel.run` becomes a real vmap-based forward path that produces a batched `ForwardState`. |
| `src/tengri/forward/forward_model.py` | Modify | `ForwardModel.predict` learns the PopulationSEDModel case: builds the per-galaxy params dict (broadcast shared, vmap per-galaxy), runs the SubModel, gets back batched state, calls `vmap(observation.predict)`. |
| `src/tengri/inference/fitter.py` | Modify | `_maybe_population_delegate` no longer dispatches; `Fitter(forward)` for a hierarchical model uses the standard `Fitter.run` path with batched data. |
| `src/tengri/inference/hierarchical.py` | Modify | `PopulationFitter` keeps working as a deprecation shim; `_via_routing` flag stays but the implementation can be replaced with a thin wrapper around the new forward path (or kept as the legacy machinery until v1.0). |
| `analysis/fig06_hierarchical_psd.py` | No change | Already uses the canonical `Fitter(forward).run` shape after PR #219; just continues working. |
| New tests | Create | `tests/contract/test_population_sed_batched_forward.py` |

---

## Task breakdown (12 tasks)

### Task 1: Decide on parameter-encoding contract (Option B above)

No code. Document the chosen per-galaxy parameter encoding in `docs/dev/archive/forward-model-architecture.md` §6.5 as a new sub-section "Hierarchical-population parameter encoding."

### Task 2: PopulationSEDModel publishes the per-galaxy-vs-shared partition

Add `PopulationSEDModel.parameter_axes()` → dict[str, int|None] returning `0` for per-galaxy axes and `None` for shared (broadcast). Used by ForwardModel.predict to construct vmap in_axes.

### Task 3: PopulationSEDModel.run uses jax.vmap

Replace the NotImplementedError stub:
```python
def run(self, state, params):
    template_run = self.sed.run
    in_axes = self._build_in_axes(state, params)
    batched = jax.vmap(template_run, in_axes=in_axes)
    return batched(state, params)
```

### Task 4: ForwardModel.predict — PopulationSED branch

Detect the case, build the shared+per-galaxy params dict, route to the vmap path. Returns `{"phot_fnu": (N, n_filters), ...}`.

### Task 5: Tests — forward path numerical equivalence

Single-galaxy fit through `ForwardModel.build(sed=sed)` vs. `ForwardModel.build(population=PopulationSEDModel(sed=sed, galaxies=[gal]))` should produce identical (up to rtol=1e-10) predictions.

### Task 6: Tests — batched forward path produces correct shapes

Build a PopulationSEDModel with N=5 galaxies, run `forward.predict(params)`, assert `phot_fnu.shape == (5, n_filters)`.

### Task 7: Fitter no longer delegates

Remove the `_population_delegate` branch in `Fitter.__init__` and `Fitter.run`. `Fitter(forward)` for a hierarchical model now needs `data` and `noise` to be batched `(N, n_filters)` arrays — the per-galaxy data on `pop.galaxies` becomes a convenience to populate those arrays.

### Task 8: Convenience: batched data from PopulationSEDModel

`PopulationSEDModel.batched_data()` → tuple of `(flux_obs, noise)` arrays of shape `(N, n_filters)`. Used by `Fitter(forward).run` to auto-construct the batched inputs.

### Task 9: PopulationFitter shim path

Either route the legacy `PopulationFitter.run` through the new forward path (one-line change with the same `PopulationPosterior` return) OR leave the legacy machinery alone and just keep the existing DeprecationWarning. Decision: keep the legacy machinery alone (Option A in design choice #4) — it works today; touching it adds risk.

### Task 10: End-to-end test — recover PSD parameters

Generate N mock galaxies with known shared PSD, run a low-iteration MAP fit through the new path, verify recovered PSD samples are within rough tolerance of truth. Single test; the analysis script remains the rigorous test.

### Task 11: Update `docs/forward_model/index.md`

The "Hierarchical population fits" section currently says inference routes through `PopulationFitter` internally. Update to say the new forward-path handles it via `jax.vmap`.

### Task 12: CHANGELOG entry + self-review + push + PR

Standard.

---

## Risk assessment

**Highest risk:** Task 3 + Task 4 — `jax.vmap` over the full SED template (stellar → dust → nebular → AGN → IGM) is non-trivial. The template's internal state involves SSP grids, filter LUTs, dust caches — each of those needs to broadcast or vmap cleanly. Suggested mitigation: incremental verification, starting with the simplest SED chain (dpl + Calzetti + none) and growing the test matrix.

**Second-highest:** Task 5 numerical equivalence at rtol=1e-10. If the vmap path produces slightly-different results (different reduction order, etc.), the test threshold needs to be eased; it's worth understanding *why* the numbers differ before doing so.

**Third:** PopulationFitter's hierarchical-sampler machinery (MGVI / EVI / raytrace specifically) is sophisticated. The new forward path inherits the underlying NUTS / VI machinery from `Fitter`, but the convergence behavior may differ — the hierarchical sampler today does some clever initialization tricks that may need to be reimplemented as `init_from=` strategies on the standard backends.

---

## Self-review checklist

- [ ] Single-galaxy `Fitter(forward).run` results identical to the existing single-galaxy path
- [ ] Multi-galaxy `Fitter(forward).run` matches `PopulationFitter.run` results (up to numerical noise + sampler differences)
- [ ] `forward.predict(params)` returns batched dict
- [ ] `analysis/fig06_hierarchical_psd.py` produces qualitatively similar figure after the routing change (with the same hyperparams)
- [ ] No new top-level entry point on `Fitter` — same `fitter.run('vi')` call drives both single-galaxy and hierarchical fits
- [ ] Existing `PopulationFitter` direct-API callers continue to work (legacy path preserved)
- [ ] DeprecationWarning surface unchanged

---

## Why this is its own PR / not bundled with #219

- The vmap path touches the SED template's internal JAX shapes — risk of breaking the (working) single-galaxy path.
- The numerical-equivalence test is the gate; until that passes at rtol=1e-10 we don't know if the new path is correct.
- Worth a focused review (the design tradeoffs above need explicit user input).
