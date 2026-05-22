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

**Wasteful-path note:** `sample(key)` is implemented as `vmap(template.spec.sample)` over a key split, which draws `(N, ...)` for *every* param — including shared ones — then overwrites shared with a single draw. This wastes N-1 RNG draws and N-1 distribution transforms per shared param. Acceptable because `sample` is only called at init time, not in the inference hot loop. Document, don't optimise.

### Task 2.5: PopulationSpecView protocol conformance test (added after review)

Add `test_population_spec_view_full_protocol_surface` — walks every attribute and method the `Fitter` actually touches on a `spec` and asserts the wrapped view returns the right shape and type. Enumerate the surface by `grep "self\.spec\." src/tengri/inference/ | grep -v test` and `grep "_spec\.\|fitter\.spec" src/tengri/inference/ | grep -v test`. This is the safety net against duck-typing leaks — a backend that reaches into a private attr like `spec._distributions` would silently break without it.

### Task 3: PopulationSEDModel.spec

`PopulationSEDModel.spec` returns a `PopulationSpecView` instance (not the bare template spec). Existing tests that compare `pop.spec` to `template.spec` need to be updated.

### Task 4: PopulationSEDModel.batched_data() helper

`pop.batched_data()` → `(flux_obs_array, noise_array)`, each shaped `(N, n_filters)`. Convenience for the user; `Fitter` can auto-call it when given `ForwardModel(population=pop)` with no `data=` / `noise=`.

### Task 5: Fitter auto-extracts batched_data

When `Fitter.__init__` receives `data=None, noise=None` AND the forward's SubModel has non-empty `batched_axes`, call `pop.batched_data()` to populate. Single-galaxy fits keep requiring explicit `data=, noise=`.

### Task 6: Remove `_maybe_population_delegate` and routing

Delete from `Fitter.__init__` and `Fitter.run`. The standard inference path now drives both single-galaxy and hierarchical fits.

### Task 6.5: Compile-cache isolation (added after review)

The new and legacy paths may share the JAX compile cache via the same `model`. If `PopulationFitter` and `Fitter(forward).run('vi')` build different JIT'd functions and cache them against the same model key, the second one to compile evicts the first — causing surprise recompiles when users switch paths. Two options:

- Each path uses a different cache key namespace (`fitter_population` vs `fitter`).
- The `PopulationFitter` deprecation path explicitly opts out of the shared cache.

Add a test (`test_compile_cache_does_not_collide_between_paths`) that runs the same hierarchical fit through both paths in succession and asserts neither rebuilds — both reuse their own cached compiles.

### Task 7: Test — numerical equivalence at MAP **and** MCMC

Two-stage gate; **MAP-only is not enough** because hierarchical samplers can drift on chain mixing even with identical $\mathcal{H}$.

**Stage 1: MAP medians at rtol=1e-4.** Generate a small mock population (N=3 galaxies); run MAP through both paths. Verify posterior medians agree within `rtol=1e-4`. This catches likelihood-broadcasting bugs.

**Stage 2: MCMC posterior percentiles.** Run NUTS through both paths with the same data, fixed key. Compute 16 / 50 / 84 percentiles for each shared parameter and a representative per-galaxy parameter. Assert they agree within MCMC noise (use the ESS-scaled standard error). This catches sampler-tuning differences (specialized init, raytrace step-size scaling) that don't show up at MAP.

If Stage 2 fails, the standard `Fitter` backend probably needs `init_from=map_result` warm-start that mimics `PopulationFitter`'s per-galaxy MAP init. Document the workaround, don't gate the PR on full chain equivalence at default settings.

### Task 7.5: psd_xi broadcasting test (added after review)

`psd_xi` (the latent stochastic-SFH field) is a special case in `_unstandardize_parameters` — it bypasses the standard `dist.unstandardize` path. Under vmap-of-`spec.sample`, `psd_xi` should naturally become shape `(N, n_xi)` — N independent SFH realisations conditional on the shared PSD. Add an explicit test that:

1. Builds a hierarchical fit with stochastic-SFH template.
2. Asserts `pop.spec.sample(key)["psd_xi"]` has shape `(N, n_xi)`.
3. Asserts the unstandardized params dict reaches `forward.predict` with the right shape (no rank confusion).

### Task 8: Test — `fitter.run('vi')` returns batched posterior

3-galaxy hierarchical fit; assert posterior samples for `sfh_dpl_alpha` have shape `(n_samples, 3)`, posterior samples for `sfh_field_psd_sigma` have shape `(n_samples,)`. Pin the batched-vs-scalar partition.

### Task 8.5: Warm-start MCMC from MAP for hierarchical (added after review)

`init_from=result_map` is the widely-used pattern: run MAP first, then NUTS / VI warm-started from the MAP point. For hierarchical, the MAP result has batched-shape per-galaxy params; `Fitter._unbounded_from_posterior` (or wherever the warm-start unbouding lives) probably assumes scalar params and will silently produce the wrong shape.

Add a test (`test_fitter_warm_start_hierarchical_map_to_nuts`) that:
1. Runs MAP on a hierarchical model.
2. Passes the result via `init_from=` to a NUTS run.
3. Verifies NUTS doesn't crash and produces samples with the right shapes.

If this fails, a small fix in the warm-start path is in scope — likely adding `axes=pop.parameter_axes(params)` handling to the unbounded-projection helper.

### Task 9: Posterior projection helpers

`Posterior.samples_for_galaxy(i)` → posterior dict for galaxy `i`. Useful diagnostic helper; no inference impact.

### Task 9.5: Mass-matrix defaults for hierarchical (added after review)

The unit-mass advantage of standardized space holds for every $\xi_i \sim \mathcal{N}(0, I)$. For hierarchical, the latent space is `(N × D_galaxy + D_hyp)`-dimensional. Dense-mass adaptation cost scales as $O((N \cdot D)^2)$ — at $N=20$, $D=137$ → $2700^2 \approx 7 \times 10^6$ elements, prohibitive.

**Action:** in `Fitter.__init__` or the NUTS backend, default mass to `diagonal` when the forward model has non-empty `batched_axes` (use the published metadata, not isinstance). Single-galaxy fits keep dense as the default for D ≲ 100. Document the heuristic.

### Task 10: Update analysis script test

`analysis/fig06_hierarchical_psd.py` already uses the new pattern. Verify it still produces qualitatively similar figure after the routing change. (Not part of CI; manual smoke test.)

### Task 11: Docs update

`docs/forward_model/index.md` "Hierarchical population fits" section currently says the inference routes through `PopulationFitter` internally. Update to: "the Fitter sees a `ForwardModel(population=...)` and runs the standard Hamiltonian path on the batched output." Reference paper §2 + §4.

### Task 11.5: Hierarchical compile-time benchmark (added after review)

Not a gate; not a hard task; just record numbers. Add a quick benchmark to `bench/scripts/`:

| Configuration | Single-galaxy compile (s) | Hierarchical compile (s) |
|---|---|---|
| Photometry, D=7 free | t_phot_single | t_phot_hier |
| Photometry, stochastic SFH D=137 | t_dense_single | t_dense_hier |

For stochastic SFH with N=20 galaxies, latent dimension is `20 × 137 + 2 = 2742`. Compile time for the gradient may be many minutes. Record so future regressions are visible.

### Task 12: CHANGELOG entry, self-review, push, PR

Standard.

---

## Risk assessment

**Highest risk:** Task 6 — removing `_maybe_population_delegate` is a breaking change for anyone who *implicitly* relied on the routing. Mitigation: Task 7's **two-stage gate** (MAP equivalence at rtol=1e-4 *and* MCMC percentile agreement within ESS-scaled noise) catches both correctness and sampler-tuning regressions. MAP-only would be insufficient.

**Second-highest:** Task 2 — `PopulationSpecView` needs to satisfy the full implicit Protocol that `Fitter` expects from `spec`. There are subtle accessors (`spec._distributions`, `spec._fixed_values`, `spec._free_names`, etc.). Task 2.5's conformance test walks the entire surface and is the safety net against duck-typing leaks.

**Third:** Task 5's auto-extraction — if the user passes `data=` and `noise=` to `Fitter(ForwardModel(population=...), data, noise)`, we should respect the explicit values (not auto-extract from `pop.galaxies`). The auto-extraction is for the "convenience default" path; explicit always wins.

**Fourth:** Task 8.5 — warm-start from MAP. The pattern `result = fitter.run('map'); fitter.run('nuts', init_from=result)` is widely used; if the unbounded-projection helper assumes scalar params, it'll silently produce wrong shapes for hierarchical. The fix is small but easy to miss without a test.

**Fifth:** Task 9.5 — dense mass-matrix default at high D. At $N=20$, $D=137$ the dense Hessian is $\sim 7\text{M}$ elements — prohibitive. Switching the default to diagonal for hierarchical fits is a one-liner if the SubModel publishes `batched_axes`. Easy to forget, expensive to discover at runtime.

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
