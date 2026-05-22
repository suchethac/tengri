# Single-Hamiltonian Fitter for hierarchical fits

> **Status:** Plan only. The largest remaining architectural piece from PR #224/#222 review. **Do not start implementation until the user reviews the four design questions below.**

**Goal:** Remove `Fitter._maybe_population_delegate` and route hierarchical PopulationSEDModel fits through the standard NUTS / HMC / VI / MAP machinery on the standardized batched output. After this lands, the inference layer has *zero* special-case code for hierarchical fits — there is **one** information Hamiltonian, evaluated on whatever shape the SubModel produces.

**Paper alignment:** Paper §2 ("Standardized Inference"): *"every Bayesian inference problem is recast as minimizing a single scalar function over a standard normal latent space."* The current `_maybe_population_delegate` branch in `Fitter.__init__` violates this — it splits inference into two paths based on SubModel type. The paper says there's supposed to be one.

**Pre-requisites (all merged):**
- PR #222: `PopulationSEDModel.run` via `jax.vmap`
- PR #224: `_predict_observation` vmaps observation when SubModel publishes `batched_axes`
- PR #237: `log_likelihood_fn` + `log_prior_fn` split on `InferenceContext`
- PR #238: end-to-end batched `forward.predict` with `.spec` delegation + `parameter_axes` rank-0 broadcast

---

## Four design questions (REQUIRES USER REVIEW)

### 1. How does `Parameters` express batched priors?

Current state: `Parameters(sfh_dpl_alpha=Uniform(0.5, 3.0))` declares one scalar prior. For a hierarchical fit, each of N galaxies needs its own draw. Three options:

**A.** **Implicit batching via `PopulationSEDModel.spec` wrapper.** Keep `Parameters` scalar-only; `PopulationSEDModel.spec` returns a *batched* view that announces "this Parameters object samples N copies of every non-shared free param." The Fitter sees `spec.free_params` exactly as today; `spec.sample(key)` returns a dict where per-galaxy params have shape `(N,)`.

**B.** **Explicit `BatchedPrior` wrapper class.** `Parameters(sfh_dpl_alpha=BatchedPrior(Uniform(0.5, 3.0), n=N))` — every per-galaxy param declares its batching at construction time. Verbose but explicit.

**C.** **Spec subclass `BatchedParameters`** that wraps the scalar spec and a population size. Same idea as A but as a named class, not a wrapped attribute.

**Recommendation: A.** The user already constructs `PopulationSEDModel(sed=template, galaxies=[...])` — the population size is known there. `template.spec` stays scalar; `pop.spec` becomes a thin wrapping view that knows N. Zero user-facing API change to `Parameters`.

### 2. How does `spec.sample(key)` produce batched draws?

If we go with A (the wrapped view):

```python
class PopulationSpecView:
    def sample(self, key):
        per_galaxy = jax.vmap(template.spec.sample)(jax.random.split(key, N))
        # per_galaxy['sfh_dpl_alpha'] shape: (N,)
        # Then overwrite shared params with one shared draw
        shared_draw = template.spec.sample(key)  # one key, one draw
        for name in self.shared:
            per_galaxy[name] = shared_draw[name]
        return per_galaxy
```

Two open questions on this:
- **How does `xi` (standardized latent) initialization work?** Today the Fitter draws `xi ~ N(0, I)` for each free param. For batched-shape params it needs `(N,)` draws. Solution: latent-shape comes from the spec's `parameter_axes` partition — done automatically by `vmap`-of-`spec.sample` if the spec is wrapped.
- **What about the cross-pop derived bundle?** `ForwardModel.predict` already namespaces derived keys for multi-pop fits (`agn.L_bol`). For hierarchical, derived keys naturally have a leading galaxy axis — no namespacing needed.

### 3. How does the likelihood get the batched data shape?

For the Fitter to consume batched data:

```python
forward = ForwardModel.build(population=pop, observation=obs)
data = jnp.stack([gal['flux_obs'] for gal in pop.galaxies])      # (N, n_filters)
noise = jnp.stack([gal['noise'] for gal in pop.galaxies])         # (N, n_filters)
fitter = Fitter(forward, data, noise)
result = fitter.run('vi')
```

Two options:

**A.** **User stacks the arrays manually.** Explicit; nothing changes in Fitter.

**B.** **`PopulationSEDModel.batched_data()` helper** returns the stacked arrays. Fitter auto-extracts when given a `ForwardModel(population=...)` with no `data=`/`noise=` argument.

**Recommendation: both.** Helper for the common case, manual stack for the escape hatch. Both produce the same `(N, n_filters)` arrays the chi² happily broadcasts over.

### 4. Migration story for `_maybe_population_delegate`

The current `_maybe_population_delegate` (PR #216) routes hierarchical fits through `PopulationFitter`. After this refactor:

**A.** **Soft cut.** `_maybe_population_delegate` is gone. The PopulationFitter direct API continues to emit a DeprecationWarning (per #219); its `model_factory` closure path is left as a back-compat shim. New code goes through `Fitter(forward).run(...)`.

**B.** **Hard cut.** Delete `PopulationFitter` entirely; replace with an alias to `Fitter`. Forces every existing call site to migrate.

**Recommendation: A.** PopulationFitter has 2000+ lines of bespoke hierarchical-sampler code (`native_vi_linear`, `evi_nifty`, `geovi`, `raytrace`). The standard Fitter path uses different inference backends and may produce numerically different (though Bayesian-equivalent) posteriors. Keeping the old path available for v0.x lets users compare results before committing to the v1.0 cut.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `src/tengri/forward/population_sed_model.py` | Modify | Add `spec` property that returns a `PopulationSpecView` (or similar) wrapping the template's spec with batched-sample knowledge. Add `batched_data()` helper. |
| `src/tengri/parameters/parameters.py` | (likely no change) | The scalar `Parameters` class stays exactly as today. |
| `src/tengri/parameters/_population_view.py` | Create | New `PopulationSpecView` (or whatever the name is). Thin wrapper over a template `Parameters` + a population size + a shared-name list. Implements the spec Protocol that `Fitter` consumes. |
| `src/tengri/inference/fitter.py` | Modify | Remove `_maybe_population_delegate` and `_population_delegate` branches. `Fitter.__init__` accepts `data=None, noise=None` only when the SubModel has empty `batched_axes` — otherwise the user must pass batched data. |
| `src/tengri/inference/loss_functions.py` | Verify (no change expected) | Confirm chi² broadcasts cleanly over the galaxy axis. The existing `_build_data_neg_log_likelihood_fn` operates on `model.predict_photometry(params)` output — once that returns `(N, n_filters)` and `data` is `(N, n_filters)`, broadcasting handles the rest. |
| `tests/contract/test_inference_entry_points.py` | Modify | Remove the routing-detection tests that pin the delegate behaviour. Add tests for the unified path. |
| `analysis/fig06_hierarchical_psd.py` | No change needed | Already uses `Fitter(forward).run(...)` after #219. The routing it goes through changes from `PopulationFitter` (today) to the standard Fitter path. Posterior shape stays the same. |
| `tests/contract/test_population_sed_model.py` | Modify | Remove tests that pin `_maybe_population_delegate` behaviour; add tests for `pop.spec` and `pop.batched_data()`. |
| `CHANGELOG.md` | Modify | "Inference layer no longer special-cases hierarchical fits..." entry. |

---

## Task breakdown (12 tasks)

### Task 1: Document the chosen design

In `docs/dev/forward-model-architecture.md`, add a §7 ("Hierarchical inference through the single Hamiltonian path") that names the recommendation chosen for each of the four questions above.

### Task 2: PopulationSpecView

Create the wrapper class. Constructor takes `template_spec, n_galaxies, shared_names`. Implements the minimum spec Protocol surface: `free_params`, `get_fixed_values`, `sample(key)`, `_distributions`, etc. Per-galaxy params get the `(N,)` shape in samples; shared params stay scalar.

### Task 3: PopulationSEDModel.spec

`PopulationSEDModel.spec` returns a `PopulationSpecView` instance (not the bare template spec). Existing tests that compare `pop.spec` to `template.spec` need to be updated.

### Task 4: PopulationSEDModel.batched_data() helper

`pop.batched_data()` → `(flux_obs_array, noise_array)`, each shaped `(N, n_filters)`. Convenience for the user; `Fitter` can auto-call it when given `ForwardModel(population=pop)` with no `data=` / `noise=`.

### Task 5: Fitter auto-extracts batched_data

When `Fitter.__init__` receives `data=None, noise=None` AND the forward's SubModel has non-empty `batched_axes`, call `pop.batched_data()` to populate. Single-galaxy fits keep requiring explicit `data=, noise=`.

### Task 6: Remove `_maybe_population_delegate` and routing

Delete from `Fitter.__init__` and `Fitter.run`. The standard inference path now drives both single-galaxy and hierarchical fits.

### Task 7: Test — numerical equivalence at MAP

Generate a small mock population (N=3 galaxies); run MAP through both paths (the legacy `PopulationFitter` and the new `Fitter(forward).run('map')`). Verify posterior medians agree within tolerance. This is the **gate** — if it fails, something about the likelihood broadcasting is wrong.

### Task 8: Test — `fitter.run('vi')` returns batched posterior

3-galaxy hierarchical fit; assert posterior samples for `sfh_dpl_alpha` have shape `(n_samples, 3)`, posterior samples for `sfh_field_psd_sigma` have shape `(n_samples,)`. Pin the batched-vs-scalar partition.

### Task 9: Posterior projection helpers

`Posterior.samples_for_galaxy(i)` → posterior dict for galaxy `i`. Useful diagnostic helper; no inference impact.

### Task 10: Update analysis script test

`analysis/fig06_hierarchical_psd.py` already uses the new pattern. Verify it still produces qualitatively similar figure after the routing change. (Not part of CI; manual smoke test.)

### Task 11: Docs update

`docs/forward_model/index.md` "Hierarchical population fits" section currently says the inference routes through `PopulationFitter` internally. Update to: "the Fitter sees a `ForwardModel(population=...)` and runs the standard Hamiltonian path on the batched output." Reference paper §2 + §4.

### Task 12: CHANGELOG entry, self-review, push, PR

Standard.

---

## Risk assessment

**Highest risk:** Task 6 — removing `_maybe_population_delegate` is a breaking change for anyone who *implicitly* relied on the routing. Mitigation: Task 7 (numerical-equivalence gate at MAP) catches regressions before they ship.

**Second-highest:** Task 2 — `PopulationSpecView` needs to satisfy the full implicit Protocol that `Fitter` expects from `spec`. There are subtle accessors (`spec._distributions`, `spec._fixed_values`, `spec._free_names`, etc.). A test that walks the entire surface (Task 2 conformance test) is the safety net.

**Third:** Task 5's auto-extraction — if the user passes `data=` and `noise=` to `Fitter(ForwardModel(population=...), data, noise)`, we should respect the explicit values (not auto-extract from `pop.galaxies`). The auto-extraction is for the "convenience default" path; explicit always wins.

---

## What this PR explicitly does NOT do

- **Likelihood broadcasting changes.** The chi² should broadcast over the galaxy axis naturally; if it doesn't, that's a separate fix surfaced by Task 7.
- **Hierarchical sampler hyperparameter tuning.** `PopulationFitter` today uses specialized init / step-size tricks for the raytrace backend. Translating those to the standard Fitter is a follow-up; the numerical-equivalence gate (Task 7) might fail without them and would tell us.
- **PopulationSEDModel.run signature changes.** Stays as-is.

---

## Self-review checklist

- [ ] `Fitter(forward).run('vi')` works identically for single-galaxy and hierarchical fits
- [ ] `Fitter._maybe_population_delegate` is completely removed from the codebase
- [ ] `PopulationSEDModel.spec` returns a batched view
- [ ] `PopulationSEDModel.batched_data()` returns `(N, n_filters)` arrays
- [ ] `analysis/fig06_hierarchical_psd.py` still runs and produces a qualitatively similar figure
- [ ] CHANGELOG entry references the single-Hamiltonian principle from paper §2
- [ ] No `if isinstance(..., PopulationSEDModel)` branches in `Fitter` or its helpers

---

## Why this is its own (substantial) PR

Removing `_maybe_population_delegate` cleanly requires the spec wrapping AND the data auto-extraction AND the numerical-equivalence gate. Each depends on the others. The biggest risk is silent numerical divergence — caught by Task 7 — which is only meaningful when the whole chain is in place.

Cannot be split into smaller PRs without one of them being "remove the working delegation, replace it with nothing temporarily" — which would break hierarchical inference in main.
