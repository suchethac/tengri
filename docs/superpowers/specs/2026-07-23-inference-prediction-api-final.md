# Inference & Prediction API — the final path

**Status:** Design spec (architecture approved in design discussion; implementation staged separately as tracked issues).
**Scope:** How an astronomer builds a model, predicts from it, and fits it — for a single galaxy, a catalog, and a hierarchical population — plus the caching, likelihood, array-shape, and approximation surfaces that support them.
**Non-goal:** This spec does **not** remove or rename any public API. It settles the *conceptual model* and the *recommended path*, and enumerates the concrete work items (some already true, most proposed) that realize it.

Legend used throughout:

- **✓ works today** — the API exists and behaves as written.
- **◆ proposed** — the shape this spec commits to; not yet built. Marked so no reader mistakes it for existing API. (This file lives under `docs/superpowers/specs/`, a design-doc location, and legitimately names not-yet-built API.)

**Tracking issues** (filed alongside this spec):

| # | Title | Work item |
|---|---|---|
| [1310](https://github.com/suchethac/tengri/issues/1310) | IGM silently dropped in WavePrecomp LUT path for non-precomputable IGM | W1b |
| [1311](https://github.com/suchethac/tengri/issues/1311) | Per-axis `FeaturePrecomp.n_grid` | — |
| [1312](https://github.com/suchethac/tengri/issues/1312) | Noisy mock predictions from a `NoiseModel` (SBI) | future |
| [1313](https://github.com/suchethac/tengri/issues/1313) | Flexibly-summarized `CatalogPosterior` for large N | future |
| [1314](https://github.com/suchethac/tengri/issues/1314) | Photo-z uncertainties in catalogs | future |
| [1315](https://github.com/suchethac/tengri/issues/1315) | `ForwardModel.build` inherit observation + LUT mismatch guard | W1 |
| [1316](https://github.com/suchethac/tengri/issues/1316) | `fit_batch(redshift_col=…)` recompiles per galaxy | W2 |
| [1317](https://github.com/suchethac/tengri/issues/1317) | Catalog API: ForwardModel + table-in/out + `CatalogPredict` + union-LUT/mask | W2 |
| [1318](https://github.com/suchethac/tengri/issues/1318) | Retire `lean=`; derive cache policy from surface; `forward.prewarm()` | W3 |
| [1319](https://github.com/suchethac/tengri/issues/1319) | Dissolve `PopulationSEDModel`/`PopulationFitter` into `ForwardModel(shared=…)` | W4 |

---

## 1. The one-paragraph mental model

Three layers, and one rule that tells you which to reach for.

```
SEDComponent      one physics block (stellar, dust, AGN, nebular, SFH-field, IGM…)
SpatialComponent  one morphology block (Sersic, point source…)
      │  compose
      ▼
SEDModel          ONE spectral composition → one SED.
                  The SIMPLE object. Physics only. Predicts standalone.
      │  compose with spatial + observation
      ▼
ForwardModel      ONE observed scene, one joint model. Combines SED + spatial +
                  multiple sub-components + observation. Output shape varies
                  (SED → cube → summed populations → hierarchical batch).
                  THE recommended surface for prediction AND inference.
      │  run many, independently
      ▼
Catalog           MANY independent ForwardModel instances, vmapped/chunked.
  ├─ CatalogPredict → mock galaxies (prediction)          ◆
  └─ CatalogFitter  → catalog posteriors (inference)      ✓ (reshaped ◆)
```

**Rule of thumb.** Eyeballing physics with no instrument → `SEDModel`. Anything you observe, fit, or that has non-trivial output shape → `ForwardModel`. Many independent galaxies → `Catalog{Predict,Fitter}`.

### Surfaces at a glance

| Surface | Predict | Fit | Output shape | Notes |
|---|---|---|---|---|
| `SEDModel` | ✓ (rest-frame; standalone photometry with a LUT) | `sed.fit()` sugar → ForwardModel ◆ | one SED | the simple object; no observation required |
| `ForwardModel` | ✓ recommended | ✓ canonical | scalar → cube → summed → hierarchical | authoritative observation |
| `CatalogPredict` ◆ | ✓ mock catalogs | — | `(N, n_filters)` | many independent, vmapped |
| `CatalogFitter` | — | ✓ `CatalogPosterior` | `(N, …)` | many independent, vmapped; table-in/out ◆ |
| `Fitter` | — | internal only | — | the cache-reuse mechanism; never taught |

---

## 2. Two axes (the decision that dissolves `PopulationSEDModel`)

"Population" previously meant two unrelated things. Separating them is the core architectural move of this spec.

| | **COMPOSE — within one scene** | **MANY — across scenes** |
|---|---|---|
| **Home** | `ForwardModel` | `Catalog{Predict,Fitter}` |
| **Is** | one generative model, one joint logdensity | N independent problems, vmapped |
| **Coupling** | parameters may be shared/coupled | none — factorizes |
| **Examples** | single galaxy · AGN+bulge+disc · multi-Sersic · **hierarchical population (shared priors)** | mock catalog · independent catalog inference |
| **Fit** | `forward.fit(...)` — one joint fit | `CatalogFitter(forward, table).run()` |
| **Could be a for-loop?** | no (coupled) | yes (catalog is just parallel) |

Two consequences that were previously conflated:

1. **Multi-component Sersic is NOT a catalog/population thing.** It is spatial composition *inside one* `ForwardModel`. One scene, one joint model.
2. **Hierarchical is ONE `ForwardModel`, not a catalog mode.** A hierarchical posterior `p(φ, θ₁…θ_N | d₁…d_N)` with shared `φ` does **not** factorize — you cannot recover it by stacking independent fits. It is a single joint model fit with the ordinary `forward.fit()`. It may use catalog-style vmap *internally*, but that is machinery, not concept. `CatalogFitter` therefore has **no** `shared=` option; a shared parameter makes it not-a-catalog.

The mathematical distinction that forces this:

```
catalog:       p(θ₁|d₁) · p(θ₂|d₂) · … · p(θ_N|d_N)      ← factorizes → embarrassingly parallel
hierarchical:  p(φ, θ₁…θ_N | d₁…d_N),  φ shared           ← does NOT factorize → one joint fit
```

`PopulationSEDModel` dissolves: its "template + galaxies + shared" role becomes a `ForwardModel` construction (`shared=…`, §6.4), fit via `forward.fit()`. `PopulationFitter` remains a deprecated shim until removal ([#1319](https://github.com/suchethac/tengri/issues/1319)).

---

## 3. Object roles and the authority rule

| Object | Owns the observation for… | Taught role |
|---|---|---|
| `SEDModel` | standalone prediction + building the SSP×filter LUT | the simple physics object; prediction; `sed.fit()` sugar |
| `ForwardModel` | **inference and recommended prediction** (authoritative) | THE surface for predict + fit; the only layer whose output shape varies |
| `Catalog{Predict,Fitter}` | inherits from the `ForwardModel` it wraps | run many independent scenes |

**Authority (settled): `ForwardModel.observation` wins.** `SEDModel` may carry an observation for standalone prediction/precompute, but at inference time the `ForwardModel`'s observation is authoritative (`forward_model.py:516,570` project through `self.observation`; `sed.observation` is never read by inference). On conflict, the LUT is reconciled to the `ForwardModel`'s filters (§5). Filters may legitimately change between the two; the only hard error is a **baked LUT used against different filters** ([#1315](https://github.com/suchethac/tengri/issues/1315)).

**Why `SEDModel` keeps the observation and is not merely a stripped physics core:** the SSP×filter LUT is an *SSP × filter* object — it cannot be built without the filters, and it must be built when the model is built. A bare `SEDModel` (no observation) is still valid — it predicts rest-frame simulation SEDs — but standalone fast photometry needs the observation for the LUT. `ForwardModel` requires an observation (mandatory today), which is exactly why no-instrument prediction lives on `SEDModel`.

---

## 4. Prediction — one contract on both surfaces

`SEDModel.predict()` and `ForwardModel.predict()` return the **same** `Prediction` object with the **same** accessors (`ForwardModel.predict` exists today at `forward_model.py:404`). Moving between them changes only which instrument-dependent accessors are available, never the idiom.

```python
# rest-frame simulation SED, no instrument — SEDModel only          ✓
sed  = SEDModel.build(ssp_data=ssp, **recipes.star_forming_photometry())
pred = sed.predict(params)                 # ONE forward pass, cached
pred.rest_sed()                            # L_nu [erg/s/Hz], rest axis
pred.rest_sed(wave)                        # resampled onto your grid
pred.properties["stellar_mass"]            # or sugar: pred.stellar_mass

# observed — recommended through ForwardModel                        ✓
fwd  = ForwardModel.build(sed=sed, observation=obs)
pred = fwd.predict(params)                 # SAME accessor contract
pred.photometry()                          # F_nu [erg/s/cm2/Hz]
pred.spectrum(wave_obs)                    # F_nu on observed grid
pred.obs_sed()                             # L_nu, obs-frame axis + IGM

# optional standalone fast prediction (SEDModel LUT)                 ✓
sed  = SEDModel.build(ssp_data=ssp, observation=obs, approx=WavePrecomp(), **cfg)
sed.predict_photometry(params)             # fast, no ForwardModel needed
```

Rules preserved from the existing prediction contract (NAMING_CONTRACT §4b):

- `predict()` takes `params` and nothing else; resampling lives on the accessor (`pred.rest_sed(wave)`).
- The SED arrays do not carry their axis — use `pred.wave_rest` / `pred.wave_obs`.
- `obs_sed()` is L_nu (a *frame*, not a flux); only `photometry()`/`spectrum()`/`magnitudes()` return a flux.
- `pred.rest_sed` without `()` raises deliberately.

---

## 5. Approximations — three LUTs, composable, built at `ForwardModel.build`

| class | accelerates | bakes | field of note |
|---|---|---|---|
| `WavePrecomp` | photometry | SSP × filter integral + redshift table | `n_z`, `z_min/z_max`, `catalog_z_range`, `n_subbands` |
| `SpectrumPrecomp` | spectroscopy | SSP × dust × IGM at spectrum pixels | `n_z`, `taylor_correction` |
| `FeaturePrecomp` | emission lines | Cue ionization grid / per-line window LUT | `n_grid`, `ranges` |

```python
sed = SEDModel.build(..., approx=WavePrecomp())                              # photometry ✓
sed = SEDModel.build(..., approx=(WavePrecomp(), FeaturePrecomp(n_grid=24))) # + lines    ✓
sed = SEDModel.build(..., approx=(WavePrecomp(n_z=200), SpectrumPrecomp()))  # joint      ✓
```

**LUT build + reuse (W1, [#1315](https://github.com/suchethac/tengri/issues/1315)).** The recommended trigger is at `ForwardModel.build(sed, obs, approx=WavePrecomp())`, against the authoritative observation, with **reuse-on-match**:

```python
fwd = ForwardModel.build(sed, obs, approx=WavePrecomp())   # ◆
#   sed carries a matching LUT         -> reuse (no rebuild)
#   sed carries a different-filter LUT -> rebuild against obs
#   sed carries no LUT                 -> build against obs
```

**Mismatch guard (W1, [#1315](https://github.com/suchethac/tengri/issues/1315)).** `observation` becomes optional on `ForwardModel.build` (inherits from the sed when omitted). The guard is scoped to the real hazard, not to object identity:

| `ForwardModel.build(sed=sed, observation=?)` | behavior |
|---|---|
| omitted | inherit `sed.observation` |
| same filters (content hash matches) | no-op |
| different filters, **no LUT** | **allowed** — filters may legitimately change; exact path recomputes |
| different filters, **LUT baked** | **raise** — LUT invalid for these filters; rebuild the sed or drop `approx` |

The filter fingerprint already exists — `SEDModel.compile_signature`'s `filter_trans_id` (`sed_model.py:3313`) is a content hash of the transmission curves — so a genuine filter *change* is distinguishable from the same filters passed twice.

**IGM regardless of LUT method (W1b, [#1310](https://github.com/suchethac/tengri/issues/1310)).** IGM must be applied on every path when it is in the model. Today the exact path applies it (`photometry.py`: `state.derived["igm_transmission"]`), and the LUT path folds a **mean** IGM into the sub-band tensor (#1135) — but a non-precomputable IGM (patchy / free-parameter) is silently absent from the LUT path. Contract: apply IGM post-LUT at projection, or raise at build if a chosen IGM variant is incompatible with the requested LUT. Never fail open.

**Per-axis `FeaturePrecomp.n_grid` ([#1311](https://github.com/suchethac/tengri/issues/1311)).** `n_grid` is currently one int for all free ionization axes (`sed_model.py:330`); the `ranges` field is already per-axis. Allow a dict for independent per-axis resolution.

---

## 6. Inference — every case, with shapes

### 6.1 Single galaxy

```python
sed = SEDModel.build(ssp_data=ssp, observation=obs, approx=WavePrecomp(), **cfg)

post = sed.fit(flux, noise, method="mcmc_nuts", key=key)   # ◆ sugar → ForwardModel
# identical, explicit surface:
fwd  = ForwardModel.build(sed=sed, observation=obs)        # ✓
post = fwd.fit(flux, noise, method="vi", key=key)          # ✓
```

**Shapes.** photometry `flux`, `noise` each `(n_filters,)`; spectroscopy `(n_pix,)`; joint via `photometry=(f,n)` and `spectrum=(f,n)`. Units `[erg/s/cm²/Hz]`. `data_type` resolves from `model.observation.data_type` (`fitter.py:648`).

`sed.fit()` is **sugar** that builds the `ForwardModel` internally. Keeping it is an *un-deprecation* of an existing method (reversing the earlier prediction-only-SEDModel framing after the Bagpipes ergonomics review) — it does not add public API, it re-blesses `sed.fit` as the astronomer one-liner while `ForwardModel.fit` stays the surface for composition/catalog/hierarchical.

### 6.2 Catalog — homogeneous (same filters): the common case

```python
fwd = ForwardModel.build(sed=sed, observation=obs,
          approx=WavePrecomp(catalog_z_range=(0.05, 1.5), n_z=200))

# INFERENCE — table in → CatalogPosterior out (CIGALE-style)  ◆ table API
cat = CatalogFitter(fwd, table, flux_cols=[...], err_cols=[...],
                    redshift_col="z").run(
          method="native_vi_linear", key=key,
          forward_chunk_size=64)                             # K vmapped per step ✓
cat["stellar_mass"]                                          # (N_galaxies,)

# PREDICTION — same noun family                               ◆ new
mock = CatalogPredict(fwd, param_table).run(chunk_size=4096)
#   → (N, n_filters) mock fluxes
```

**Known redshifts from a catalog are first-class.** Redshift is a *column*; `catalog_z_range` makes the whole catalog **one compile** (`sed_model.py:188-212`), each row's `z` flowing in as a runtime value (~µs ztable interpolation) instead of one compile per row. Photo-z *uncertainties* are the future extension in [#1314](https://github.com/suchethac/tengri/issues/1314) (`redshift_err_col` / `redshift_pz`).

**Shapes.** internally stacks to contiguous `(N_galaxies, n_filters)` for flux and noise, `(N_galaxies,)` for redshift (see §9 for the materialization contract).

**Scaling knobs (CompoSED-parity).** `forward_chunk_size=K` (vmap K per `lax.map` step; XLA graph O(1) in N — `catalog_fitter.py`); `n_pad="auto"` (bucket catalog sizes to reuse one compile).

**The `fit_batch` recompile cliff (W2, [#1316](https://github.com/suchethac/tengri/issues/1316)).** Today `fit_batch(redshift_col=…)` clones a fresh `SEDModel` per row (`convenience.py:510-520`); since `z_fixed` is in `compile_signature`, that is a new instance *and* a new signature → a full recompile per galaxy, silently. The reshaped catalog surface must auto-enable `catalog_z_range` from the catalog's z span, or raise/warn if `approx` lacks it.

### 6.3 Catalog — heterogeneous (different filters per galaxy)

Contract: **a galaxy filter not in the LUT set → raise.** Two paths ([#1317](https://github.com/suchethac/tengri/issues/1317)):

```python
# fallback that works today — sequential, per-galaxy compile     ✓
CatalogFitter(fwd, galaxies).run(method="map", forward_chunk_size=1)
#   forward_chunk_size=1 → no vmap → ragged n_data allowed
#   (forward_chunk_size>1 currently raises on ragged grids:
#    _validate_homogeneous_galaxies, population_sed_model.py:87)

# target — vmapped via union-LUT + mask                          ◆
approx = WavePrecomp(filters=union_filter_set,                   # ◆ explicit union
                     catalog_z_range=(0.05, 1.5))
fwd    = ForwardModel.build(sed=sed,
             observation=Observation(filters=union_filter_set), approx=approx)
cat = CatalogFitter(fwd, table, flux_cols=UNION_COLS, err_cols=UNION_ERRS,
                    redshift_col="z").run(method="native_vi_linear",
                    key=key, forward_chunk_size=64)
```

**Shapes (union path).** rectangular over the union: `flux`, `noise`, and a **mask** each `(N_galaxies, n_union)`. A galaxy missing band *b* has `mask[i,b]=False`; χ² skips it (contributes 0). The model predicts all `n_union` bands from the single union LUT; the mask selects per galaxy. The single-galaxy mask seam already exists (`Fitter(..., data_mask=…)`); the ◆ work lifts it to a `(N, n_union)` catalog mask plus the union-LUT builder. A band a galaxy *has* but the union *lacks* → **raise**.

### 6.4 Hierarchical — one joint model, internal vmap (W4, [#1319](https://github.com/suchethac/tengri/issues/1319), deferred)

```python
fwd = ForwardModel.build(
    sed=template_sed, observation=obs,
    shared=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"),   # ◆ declares joint structure
    approx=WavePrecomp(catalog_z_range=(0.05, 1.5)))

post = fwd.fit(table, flux_cols=[...], err_cols=[...], redshift_col="z",
               method="vi", key=key)          # ONE joint fit over all N
post.shared_samples["sfh_field_psd_sigma"]    # population hyperparameter
post.properties["stellar_mass"]               # per-galaxy, (N,)
```

**Shapes.** data `(N_galaxies, n_filters)` (homogeneous grid required by the shared-observation contract). The parameter vector is `{φ_shared} ∪ {θ_i}` — one joint vector. The forward vmaps the template over galaxies internally (`lax.map(batch_size=K)`); conceptually it is a single model.

---

## 7. Defining the likelihood — a config, not a subclass

The likelihood never appears in the sampling loop the astronomer sees. They pick a `NoiseModel` (`observation/noise_model.py`); the machinery wires it.

```python
from tengri import NoiseModel, Uniform

NoiseModel()                                    # diagonal Gaussian (default)     ✓
NoiseModel(calibration_floor=0.05)              # fixed cal floor in quadrature   ✓
NoiseModel(calibration_floor=Uniform(0.01, 0.15))  # cal floor as a free param   ✓
NoiseModel(student_t_dof=10)                    # heavy-tailed, outlier-robust    ✓

# spectroscopy: marginalize a polynomial flux calibration (at fit time)          ✓
fwd.fit(flux, noise, calibration_marginalize=True, cal_n_poly=3)
```

`sigma_eff = sqrt(sigma_obs² + (f_cal · model)²)`, optionally Student-t. Escape hatch — the `Likelihood` protocol (`protocols/likelihood.py`): `log_prob(prediction, data, noise_params) -> scalar`, `declared_parameters()`, `name`. A custom GP or composite likelihood is `fwd.fit(..., likelihood=MyLikelihood())`. ✓

This binding is CompoSED's `Problem`, minus one object: the `ForwardModel` already carries the priors (`spec`) and instrument (`observation`); `data`, `noise`, and the `NoiseModel` arrive at `fit()`.

```
CompoSED:   Problem(backend, parameters, data, likelihood, filters); fit(problem, sampler)
tengri:     ForwardModel(sed + observation + priors-in-spec);        fwd.fit(data, noise, NoiseModel, method)
                        └──────────── "the problem" ───────────┘
```

**Noisy mock draws for SBI (future, [#1312](https://github.com/suchethac/tengri/issues/1312)).** `NoiseModel` today configures the *likelihood*; `fwd.simulate(params, noise=…, key=…)` (and the batched `CatalogPredict(...).simulate(...)`) draws *noisy* observations using the **same** `sigma_eff` the likelihood uses, closing the simulate/fit loop for amortized inference.

---

## 8. Caching — three tiers, made legible

The split is finer than model-vs-inference. Ask of any compiled artifact: *does it depend on the galaxy's actual flux values?*

| Tier | Depends on | Reusable across | Artifacts |
|---|---|---|---|
| **1 Physics** | model structure only | everything (all data, galaxies, methods, backends) | `signal_response`, structural prediction kernels, the SSP×filter LUT |
| **2 Problem shape** | + data **shape**, free-param set, feature channels — **not values** | all same-shape galaxies | `loss_fn`, `grad_fn`, `logdensity_fn` |
| **3 This fit** | + data **values**, method, kwargs | nothing | NUTS adaptation, mass matrix, MAP result |

Tier 2 is galaxy-agnostic *by construction* — `data_args` is a **traced** argument, never closed over (`backends/mcmc/_shared.py:25`) — which is exactly why `CatalogFitter` can vmap one compiled logdensity over N data rows. This is the analogue of CompoSED's "train once, apply to 100k", except tengri's is **exact** rather than amortized.

All expensive caches are keyed on the **model** (a `WeakKeyDictionary` in `inference/_model_cache.py`), not on any `Fitter`. A fresh `Fitter` on the same model hits the same warm cache — which is why `Fitter` has no reason to be a user-facing noun (§3).

**Reuse and prewarm (W3, [#1318](https://github.com/suchethac/tengri/issues/1318)).** Reuse is automatic: repeated `forward.fit()` on the same `ForwardModel` reuses tiers 1+2. Advanced opt-in: `forward.prewarm()` builds tiers 1+2 ahead of time (today `prewarm` exists on the internal `Fitter` only). `Fitter` stays internal; it never appears in taught docs.

**`lean` retired (W3, [#1318](https://github.com/suchethac/tengri/issues/1318)).** The tier-3 eviction policy (`fitter.py:2068-2103`) stops being a mechanism-named kwarg on `run()`. Intent is expressed by the *surface*: `forward.fit()` uses an iterate-friendly policy (keep everything); `CatalogFitter`/`CatalogPredict` use a sweep policy (keep tiers 1+2, drop per-galaxy tier-3 automatically). `lean=True/False` remains a hidden deprecated alias.

---

## 9. Array shapes and memory — the contract

Two independent efficiency levers, each a firm requirement.

### 9.1 Input — always materialize contiguous

**Requirement.** The catalog surfaces accept **either** table-in (a table + column names, astronomer-friendly) **or** arrays-in (pre-stacked, power user), but **internally always materialize contiguous arrays** before entering any JAX transform:

- `flux`  → `(N, n_data)` contiguous
- `noise` → `(N, n_data)` contiguous
- `redshift` → `(N,)` contiguous
- `mask` (heterogeneous only) → `(N, n_union)` boolean

That is the shape vmap wants anyway. The `list[dict]` form (today's `CatalogFitter`, `catalog_fitter.py:272`) is accepted but immediately stacked; it must never be the internal representation on the hot path.

| input form | memory | notes |
|---|---|---|
| `list[dict]` | worst | per-galaxy Python/array-object overhead; N small device arrays; N host→device transfers |
| table + column names | good | columns already contiguous; one `(N, n_data)` stack |
| pre-stacked `(N, n_data)` arrays | best | one contiguous device array, one transfer, vmap-native |

At N=10⁵, n_data=10, `list[dict]` carries hundreds of MB of overhead over the ~16 MB of actual numbers — ~10× more memory and N× more transfers than the contiguous stack.

### 9.2 Output — flexible summary, not just fixed quantiles

**Requirement.** For large N, `CatalogPosterior` must support **streaming, flexibly-summarized** storage ([#1313](https://github.com/suchethac/tengri/issues/1313)) — process the catalog in chunks, reduce each chunk, discard the sample cube. The summary is **configurable**, not a fixed quantile triple:

- **arbitrary percentiles** — caller supplies the list (e.g. `(2.5, 16, 50, 84, 97.5)`).
- **custom reducers** — arbitrary summary callables (mean, std, MAP, mode, HPD interval, user `samples -> summary`), composable with the percentiles.
- **`store="full"`** stays available for exploratory small-N work.

```python
cat = CatalogFitter(fwd, table, flux_cols=[...], err_cols=[...]).run(
    method="native_vi_linear", key=key, forward_chunk_size=64,
    store="summary",                                  # vs "full"; default depends on N
    percentiles=(2.5, 16, 50, 84, 97.5),              # flexible, caller-chosen
    reducers={"mean": jnp.mean, "std": jnp.std},      # + arbitrary custom reducers
)
cat.percentiles["stellar_mass"]     # (N, len(percentiles))
cat.summary["mean"]["stellar_mass"] # (N,)
cat["stellar_mass"]                 # median convenience, (N,)
```

An MCMC catalog storing the full `(N, n_samples, n_params)` cube is ~8 GB at N=10⁵. The reduction happens at the `forward_chunk_size` chunk boundary. Default `store="full"` below a threshold, `store="summary"` above it — **log the switch, never silently truncate**.

---

## 10. What peer codes taught us

- **Bagpipes** — nested-dict components + a one-liner fit. tengri matches the dict (its grammar) and re-blesses `sed.fit()` for the one-liner.
- **CIGALE** — catalog-native, table-in/table-out, redshift as a column. Adopted wholesale for `CatalogFitter`/`CatalogPredict`.
- **CompoSED** — the likelihood is a thin config (`Gaussian(sigma_floor=…)`); catalog batching is transparent with chunk-size knobs; and — the headline — large-N throughput comes from never storing the sample cube. Adopted as `NoiseModel`, `forward_chunk_size`/`n_pad`, and the flexible-summary [#1313](https://github.com/suchethac/tengri/issues/1313).
- **Prospector** — the anti-patterns: redshift buried in an `obs` dict, and build-functions the user must write. Avoided: redshift is a top-level key / column; construction is `build`-classmethods with recipes.

---

## 11. Work items

| ID | Piece | Status | Tracks |
|---|---|---|---|
| — | single/catalog/hierarchical construct + `predict`/`fit`; one predict contract | ✓ | — |
| **W1** | `observation` optional on `ForwardModel.build` + inherit + LUT reuse-on-match + LUT-scoped mismatch guard | ◆ | [#1315](https://github.com/suchethac/tengri/issues/1315) |
| **W1b** | IGM applied on every LUT path (or raise) | ◆ | [#1310](https://github.com/suchethac/tengri/issues/1310) |
| **W2** | `fit_batch`/catalog recompile cliff (auto `catalog_z_range` or warn) | ◆ | [#1316](https://github.com/suchethac/tengri/issues/1316) |
| **W2** | Catalog reshape: take a `ForwardModel`, table-in/out, `(N, n_union)` mask + union-LUT + subset-error, `CatalogPredict` | ◆ | [#1317](https://github.com/suchethac/tengri/issues/1317) |
| **W3** | `forward.prewarm()`; retire `lean`, derive tier-3 policy from surface | ◆ | [#1318](https://github.com/suchethac/tengri/issues/1318) |
| **W4** | hierarchical as a `ForwardModel` construction (`shared=`); `PopulationSEDModel`/`PopulationFitter` dissolve | ◆ deferred | [#1319](https://github.com/suchethac/tengri/issues/1319) |
| — | per-axis `FeaturePrecomp.n_grid` | ◆ | [#1311](https://github.com/suchethac/tengri/issues/1311) |
| — | flexibly-summarized `CatalogPosterior` (percentiles + reducers) | ◆ future | [#1313](https://github.com/suchethac/tengri/issues/1313) |
| — | noisy mock draws (`simulate`) for SBI | ◆ future | [#1312](https://github.com/suchethac/tengri/issues/1312) |
| — | photo-z uncertainties in catalogs | ◆ future | [#1314](https://github.com/suchethac/tengri/issues/1314) |

---

## 12. Decisions log

Chronological record of the calls made during design, with rationale — so a future reader knows *why*, not just *what*.

1. **Public API unchanged; `ForwardModel.fit` canonical.** Reverses nothing shipped; keeps #211 direction for the composition/catalog surface.
2. **`SEDModel` keeps its observation.** It is what makes the SSP×filter LUT buildable and lets a bare `SEDModel` predict simulation SEDs with no fitting. (Measured: `sed_model.py:171-173`.)
3. **Authority: `ForwardModel.observation` wins.** Inference and recommended prediction go through it; the sed's observation is for standalone use. (Measured: nothing reconciles the two today — the silent-mismatch hole → W1.)
4. **SEDModel = simple / ForwardModel = shape-general.** The boundary is *output-shape complexity*, not merely presence of an instrument. ForwardModel is the one layer whose output shape varies.
5. **One predict contract** across both surfaces (already ~true; `forward_model.py:404`).
6. **Catalog stays a separate fast path**, takes a `ForwardModel`, table-in/out; `CatalogPredict` is its prediction twin.
7. **Hierarchical is ONE `ForwardModel`, not a catalog mode.** The joint posterior does not factorize; `CatalogFitter` has no `shared=`. (Corrected mid-discussion from the looser "catalog with shared params" framing.)
8. **`sed.fit()` kept as sugar.** After the Bagpipes ergonomics review, the astronomer one-liner won; it is an un-deprecation, not new API.
9. **`Fitter` internal.** All expensive caches are model-keyed, so `Fitter` has no state a fresh one lacks; it is the cache-reuse mechanism, never taught.
10. **`lean` retired to a surface-derived policy**; `forward.prewarm()` exposed.
11. **Input always contiguous; output flexibly summarized** (percentiles + reducers), not fixed quantiles.
12. **Deliverable is this spec; implementation is the tracked issues.**

---

## 13. Non-goals / explicit compatibility

- **No public API is removed.** `Fitter`, `CatalogFitter`, `PopulationFitter` stay importable; `sed.fit` is un-deprecated, not added.
- **`ForwardModel.fit` remains the canonical inference surface.** `sed.fit` is sugar over it.
- **`Fitter` stays internal.** It is the cache-reuse mechanism, never taught.
- **The catalog↔population unification (W4) is deferred.** Recorded so it is not lost; out of scope for Paper I.
- **The deep `InferenceContext`/backend decoupling (ADR-0010) is out of scope here.** It is pure internal hygiene with no user-visible payoff and does not gate any decision in this spec.
