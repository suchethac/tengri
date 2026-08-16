# Inference & Prediction API — the final path

**Status:** Design spec v2 (architecture + review round complete; implementation staged as tracked issues).
**Scope:** How an astronomer builds a model, predicts from it, and fits it — single galaxy, catalog, hierarchical population — plus the observation/data split, likelihood, caching, array-shape, and approximation surfaces that support them.
**Non-goal:** This spec removes no public API. It settles the *conceptual model* and the *recommended path*, and enumerates the work items that realize it.

Legend:

- **✓ works today** — the API exists and behaves as written.
- **◆ proposed** — the shape this spec commits to; not yet built. (This file lives under `docs/internal/specs/`, a design-doc location, and legitimately names not-yet-built API.)

**Tracking issues.** Status reconciled against `main` on 2026-08-05; the ✓/◆ markers
throughout this document follow this table. Reconciliation is against **behavior on
`main`**, not against issue-closure state — an issue can stay open on a residual while
the API it describes is shipped and taught, and (as #1344 showed) a closed issue can
leave half its spec unbuilt.

| # | Title | Work item | Status |
|---|---|---|---|
| [1310](https://github.com/suchethac/tengri/issues/1310) | ~~IGM silently dropped in WavePrecomp LUT path~~ **closed stale** (already fixed by #1135/#1149; verified by execution) | W1b | ✓ closed stale |
| [1329](https://github.com/suchethac/tengri/issues/1329) | `fit(params=...)` silently swallowed — the WavePrecomp docstring advertises a kwarg nothing consumes | W2 dep | ✓ shipped |
| [1311](https://github.com/suchethac/tengri/issues/1311) | Per-axis `FeaturePrecomp.n_grid` | — | ✓ shipped |
| [1312](https://github.com/suchethac/tengri/issues/1312) | Noisy mock predictions (`simulate`) for SBI | future | ◆ open |
| [1313](https://github.com/suchethac/tengri/issues/1313) | Flexibly-summarized `CatalogPosterior` (percentiles + reducers) | future | ✓ shipped |
| [1314](https://github.com/suchethac/tengri/issues/1314) | Photo-z uncertainties in catalogs | future | ◆ open |
| [1315](https://github.com/suchethac/tengri/issues/1315) | `ForwardModel.build` inherits observation + LUT mismatch guard | W1 | ✓ shipped |
| [1316](https://github.com/suchethac/tengri/issues/1316) | `fit_batch(redshift_col=…)` recompiles per galaxy | W2 | ✓ shipped |
| [1317](https://github.com/suchethac/tengri/issues/1317) | `Catalog`: one noun, table-in/out, name-matching, union-LUT + presence mask | W2 | ✓ shipped |
| [1318](https://github.com/suchethac/tengri/issues/1318) | Retire `lean=`; surface-derived cache policy; `forward.prewarm()` | W3 | ✓ shipped |
| [1319](https://github.com/suchethac/tengri/issues/1319) | Hierarchical as `ForwardModel(mode="hierarchical", shared=…)`; Population classes dissolve | W4 | ◆ two-track (decision 23): standardization a must; realistic fits two-step ([PR #1479](https://github.com/suchethac/tengri/pull/1479) ✓) |
| [1321](https://github.com/suchethac/tengri/issues/1321) | Observation = pure instrument; introduce `Data` record | W5 | ✓ shipped |
| [1395](https://github.com/suchethac/tengri/issues/1395) | `sfh_model="table"` slips past the fast-path guard → zero SFH, zero lines, no warning | W6 blocker | ✓ shipped (now raises) |
| [1396](https://github.com/suchethac/tengri/issues/1396) | `Catalog.from_histories` + `simulate`: mock catalogs from simulation SFH/Z tables | W6 | ✓ shipped, round-trip test included (see §8.1 for residuals) |
| [1522](https://github.com/suchethac/tengri/issues/1522) | Tabulated SFH silently dropped mass older than the SSP grid's oldest age | W6 defect | ✓ fixed (mass conserved; color approximation now warns) |

---

## 1. The one-paragraph mental model

Three layers, one runner, and one rule that tells you which to reach for.

```
SEDComponent      one physics block (stellar, dust, AGN, nebular, SFH-field, IGM…)
SpatialComponent  one morphology block (Sersic, point source…)
      │  compose
      ▼
SEDModel          ONE spectral composition → one SED.
                  The SIMPLE object. Physics only. Predicts standalone.
      │  compose with spatial + observation (the instrument schema)
      ▼
ForwardModel      ONE observed scene, one joint model. Combines SED + spatial +
                  multiple sub-components + observation. Output shape varies
                  (SED → cube → summed populations → hierarchical batch).
                  THE recommended surface for prediction AND inference.
      │  run many, independently
      ▼
Catalog           MANY independent ForwardModel problems, vmapped/chunked.  ✓
                  One noun, action verbs: .fit() ✓ / .predict() ✓ / .simulate() ✓
```

**Rule of thumb.** Eyeballing physics with no instrument → `SEDModel`. Anything you observe, fit, or that has non-trivial output shape → `ForwardModel`. Many independent galaxies → `Catalog`.

**The uniform data rule** (§3): *models never hold measured values.* Data enters at the **action** — `fit()`, `predict()`, `Catalog(...)` — never at model construction.

### Surfaces at a glance

| Surface | Predict | Fit | Output shape | Notes |
|---|---|---|---|---|
| `SEDModel` | ✓ (rest-frame; standalone LUT photometry) | `sed.fit()` sugar → ForwardModel ✓ | one SED | the simple object; no observation required to exist |
| `ForwardModel` | ✓ recommended | ✓ canonical | scalar → cube → summed → `(N, …)` | authoritative observation; `mode=` inferred or asserted ✓ |
| `Catalog` ✓ | `.predict()` mocks ✓ · `.simulate()` ✓ | `.fit()` → `CatalogPosterior` ✓ | `(N, …)` | one noun; `CatalogFitter` is a deprecated alias |
| `Fitter` | — | internal only | — | the cache-reuse mechanism; never taught |

---

## 2. Two axes (the decision that dissolves `PopulationSEDModel`)

"Population" previously meant two unrelated things. Separating them is the core architectural move.

| | **COMPOSE — within one scene** | **MANY — across scenes** |
|---|---|---|
| **Home** | `ForwardModel` | `Catalog` |
| **Is** | one generative model, one joint logdensity | N independent problems, vmapped |
| **Coupling** | parameters may be shared/coupled | none — factorizes |
| **Examples** | single galaxy · AGN+bulge+disc · multi-Sersic · **hierarchical population (shared priors)** | mock catalog · independent catalog inference |
| **Fit** | `forward.fit(...)` — one joint fit | `Catalog(fwd, table).fit(...)` |
| **Could be a for-loop?** | no (coupled) | yes (catalog is just parallel) |

The mathematics that forces this:

```
catalog:       p(θ₁|d₁) · p(θ₂|d₂) · … · p(θ_N|d_N)      ← factorizes → embarrassingly parallel
hierarchical:  p(φ, θ₁…θ_N | d₁…d_N),  φ shared           ← does NOT factorize → one joint fit
```

Consequences previously conflated:

1. **Multi-component Sersic is NOT a catalog/population thing.** It is spatial composition *inside one* `ForwardModel`. One scene, one joint model.
2. **Hierarchical is ONE `ForwardModel`, not a catalog mode.** Shared `φ` couples every galaxy — you cannot recover the joint posterior by stacking independent fits. It may use catalog-style vmap *internally* (§6.4 scaling contract), but that is machinery, not concept. `Catalog` therefore has **no** `shared=` option.

`PopulationSEDModel` dissolves into `ForwardModel.build(mode="hierarchical", shared=…)` (§6.4); `PopulationFitter` remains a deprecated shim until removal ([#1319](https://github.com/suchethac/tengri/issues/1319)).

---

## 3. The data-free rule: Observation is schema, Data is record

**The razor: an `Observation` holds everything true *before you point the telescope*** — filters, gratings, wavelength grid, LSF, noise character, and *which* lines/indices you intend to measure. **A `Data` holds what came back** — fluxes, errors, limits. ([#1321](https://github.com/suchethac/tengri/issues/1321))

Today's `Observation` is a hybrid: `photometry`/`spectroscopy`/`noise` are instrument config, but `line_fluxes`/`spectral_indices`/`line_ratios` hold **measured values**. Fitting two galaxies with different measured line fluxes therefore needs two Observations → two ForwardModels → recompiles. The engine cache key already knows the razor — it contains the line *wavelengths* (definition) but not their values; the API just never followed suit.

### 3.1 Observation — the instrument schema ✓ (razor applied; all constructors ✓)

None of the instrument flexibility changes — it all lives in `Photometry`/`Spectroscopy` already:

```python
# one survey                                                          ✓
obs = Observation(photometry=Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]))

# combination of surveys — the registry has no survey boundaries      ✓
obs = Observation(photometry=Photometry.from_names(
    ["galex_fuv", "galex_nuv", "sdss_g", "sdss_r", "sdss_i",
     "2mass_j", "2mass_ks", "wise_w1", "wise_w2"]))
# 249 curves ship in data/filters/; unknown names fetch from SVO      ✓

# brand-new filter set — your own curves                              ✓
obs = Observation(photometry=Photometry.from_filter_set(my_curves))

# spectroscopy: a bare wavelength array is enough                     ✓
Spectroscopy(wave_obs=wave)                       # no LSF
Spectroscopy(wave_obs=wave, resolution=1000.0)    # scalar R
Spectroscopy(wave_obs=wave, resolution=R_array)   # R per pixel = LSF(λ), validated to n_pix
Spectroscopy(wave_obs=wave, resolution_matrix=M)  # full banded matrix (DESI-style)

# the full schema                                                     ✓ (lines= is the razor change)
obs = Observation(
    photometry   = Photometry.from_names(["jwst_f200w", "jwst_f356w"]),
    spectroscopy = Spectroscopy.nirspec_prism(wave_obs),
    noise        = NoiseModel(calibration_floor=Uniform(0.01, 0.15)),   # instrument noise character
    lines        = LineList.from_names(["Halpha", "OIII_5007"]),        # WHICH lines — not fluxes
)
```

`line_fluxes`/`spectral_indices`/`line_ratios` keep working with a one-shot `DeprecationWarning`; their *definitions* stay in the schema, their *values* move to `Data`.

### 3.2 Data — the measurement record ✓ (adopted; shipped in W5)

`Data` is a small frozen container validated against the model's Observation at `fit()` — **one seam** for shape checks, NaN policy, censor alignment, and line-name subsetting:

| Observation declares (schema) | Data supplies (record) |
|---|---|
| `photometry` — n named filters | `photometry=(flux, err)` — each `(n_filters,)` |
| `spectroscopy` — wave grid, LSF | `spectrum=(flux, err)` — each `(n_pix,)` |
| `lines=LineList.from_names([...])` — *which* lines | `lines={"Halpha": (val, err)}` — their fluxes |
| `noise=NoiseModel(...)` — noise *character* | `censor=flags` — this galaxy's limits `0/1/-1` |

```python
data = Data(photometry=(flux, err),
            spectrum=(spec_flux, spec_err),
            lines={"Halpha": (3.2e-17, 0.4e-17)},
            censor=phot_censor_flags)            # optional
fwd.fit(data, method="vi", key=key)              # multi-channel
fwd.fit(flux, err, method="vi", key=key)         # sugar: photometry-only, unchanged ✓
```

**Schema : record : table.** A `Catalog` table is simply **N records of the same schema** — same validation, vectorized. Which is exactly why one instrument + a table can vmap: same schema ⇒ same compiled program; values stream through as traced inputs.

**The uniform rule this completes:** models never hold measured values. Not `Observation` (schema), not `ForwardModel` (even hierarchical — §6.4). Data enters at the action: `fit` / `predict` / `Catalog`. N is inferred at the action too — from the table at `Catalog`, from the data batch at a hierarchical `fit`, from the params' leading axis at a batched `predict`.

### 3.3 Missing vs censored — never the same channel

Two concepts that must never share a representation:

- **Censored** (`censor` flags, per band): a measurement *exists* and is informative — `0` = detected (Gaussian term), `1` = upper limit (`ln Φ((limit−model)/σ)`), `−1` = lower limit. This is today's `data_mask` (`fitter.py:299-306`), feeding `CensoredLikelihood`. **Boolean arrays are rejected by design** — `True` would silently read as "upper limit".
- **Absent** (presence mask, Catalog union path only, §6.3): no measurement exists for this galaxy in this band — χ² skips it entirely.

A single-galaxy `Data` must be *complete* with respect to its schema — if you did not observe a band, remove the filter from the Observation. *Absence* exists only on the Catalog union path, where one shared schema spans galaxies with different coverage.

---

## 4. Object roles and the authority rule

| Object | Holds | Taught role |
|---|---|---|
| `SEDModel` | physics (+ optional observation for standalone LUT/predict) | the simple object; prediction; `sed.fit()` sugar |
| `Observation` | the instrument schema (compile-relevant, data-free) | declared once; validated against every `Data` |
| `ForwardModel` | scene composition + **authoritative** observation | THE surface for predict + fit |
| `Data` | one galaxy's measurements | created at the action |
| `Catalog` | a ForwardModel + N records | run many independent scenes |

**Authority (settled): `ForwardModel.observation` wins.** `SEDModel` may carry an observation for standalone prediction/precompute, but at inference time the ForwardModel's observation is authoritative (`forward_model.py:516,570` project through `self.observation`; the sed's is never read by inference). With the razor in place this rule is unambiguous — instrument *config* can be authoritative; data is per-galaxy and never lives on a model. On conflict, the LUT is reconciled to the ForwardModel's filters (§5, [#1315](https://github.com/suchethac/tengri/issues/1315)).

**`SEDModel` needs no observation to exist** — a bare `SEDModel` predicts rest-frame simulation SEDs. `ForwardModel` requires one (it *is* the observed-scene layer), which is exactly why no-instrument prediction lives on `SEDModel`.

**`mode=` on `ForwardModel.build` ✓ (adopted; shipped): inferred by default, assertable explicitly.**

```python
ForwardModel.build(sed=sed, observation=obs)                      # → inferred "single"
ForwardModel.build(sed=template, observation=obs,
                   shared=("sfh_field_psd_sigma",))               # → inferred "hierarchical"
ForwardModel.build(populations=[...], observation=obs)            # → inferred "multi_population"

ForwardModel.build(mode="hierarchical", sed=template,
                   observation=obs, shared=(...))                 # asserted + validated
# mode given but its required kwargs missing → ONE mode-aware error
# mode="single" with shared=              → error: shared is hierarchical-mode
```

Novices never type `mode=`; explicit users get early, precise validation instead of scattered kwarg errors.

---

## 5. Approximations — three LUTs, composable, taught at `ForwardModel.build`

| class | accelerates | bakes | key fields |
|---|---|---|---|
| `WavePrecomp` | photometry | SSP × filter integral + redshift table | `n_z`, `z_min`, `z_max`, `catalog_z_range`, `n_subbands`, `taylor_correction`, `fast_dust_emission` |
| `SpectrumPrecomp` | spectroscopy | SSP × dust × IGM at spectrum pixels | `n_z`, `taylor_correction` |
| `FeaturePrecomp` | emission lines | Cue ionization grid / per-line window LUT | `n_grid`, `ranges` |

```python
fwd = ForwardModel.build(sed=sed, observation=obs, approx=WavePrecomp())          # ✓ taught placement
fwd = ForwardModel.build(..., approx=(WavePrecomp(), FeaturePrecomp(n_grid=24)))  # + lines
fwd = ForwardModel.build(..., approx=(WavePrecomp(n_z=200), SpectrumPrecomp()))   # joint
# SEDModel.build(..., approx=...) stays as the standalone-prediction opt-in        ✓
```

**LUT build + reuse (W1, [#1315](https://github.com/suchethac/tengri/issues/1315)) ✓.** Built at `ForwardModel.build` against the authoritative observation, with reuse-on-match: sed carries a matching LUT → reuse; different-filter LUT → rebuild; no LUT → build.

**Mismatch guard (W1) ✓.** `observation` is optional on `ForwardModel.build` (inherits from the sed when omitted). The guard is scoped to the real hazard:

| `ForwardModel.build(sed=sed, observation=?)` | behavior |
|---|---|
| omitted | inherit `sed.observation` |
| same filters (content hash matches) | no-op |
| different filters, **no LUT** | **allowed** — filters legitimately change; exact path recomputes |
| different filters, **LUT baked** | **raise** — LUT invalid; rebuild the sed or drop `approx` |

The fingerprint exists: `compile_signature`'s `filter_trans_id` (`sed_model.py:3313`) content-hashes the transmission curves — a genuine filter change is distinguishable from the same filters passed twice.

**IGM regardless of LUT method (W1b — resolved; [#1310](https://github.com/suchethac/tengri/issues/1310) closed stale).** The contract — IGM applies on every path, never fails open — **already holds**: the exact path applies `state.derived["igm_transmission"]`, and the #1135/#1149 sub-band fold covers the LUT path including patchy IGM. Verified 2026-07-23 by *executing* the path: 28/28 tests across `test_bug_1149_patchy_igm_jit.py`, `test_bug_1135_igm_subband_precompute.py`, and `test_igm_reaches_photometry.py` pass on main. An earlier revision of this spec asserted a silent drop for non-precomputable IGM; that claim came from reading a narrow code comment rather than executing the path, and was wrong. The contract stays stated here so any future LUT variant is held to it.

**Per-axis `FeaturePrecomp.n_grid` ([#1311](https://github.com/suchethac/tengri/issues/1311)).** `n_grid` is one int for all free ionization axes; `ranges` is already per-axis. Allow a dict.

---

## 6. Inference — every case, with shapes

### 6.1 Single galaxy

```python
sed = SEDModel.build(ssp_data=ssp, observation=obs, **cfg)
fwd = ForwardModel.build(sed=sed, approx=WavePrecomp())        # obs inherited ✓

post = fwd.fit(flux, err, method="mcmc_nuts", key=key)         # ✓ (bare-array sugar)
post = fwd.fit(data, method="vi", key=key)                     # ✓ single-channel Data record
post = sed.fit(flux, err, method="vi", key=key)                # ✓ astronomer one-liner (sugar → ForwardModel)
```

**Shapes.** photometry `(n_filters,)`; spectroscopy `(n_pix,)`; joint & censored via `Data`. Units: arrays are cgs `[erg/s/cm²/Hz]` (documented); unit declaration is a table-ingestion concern (§6.2). `data_type` resolves from the observation (`fitter.py:648`).

> **Joint channels.** A single-channel `Data` record works; the *joint*
> `Data(photometry=…, spectrum=…)` record raises on this surface until
> [#1393](https://github.com/suchethac/tengri/pull/1393) lands. Photometry-only
> records passing is why the gap survived its own tests.

**Censored bands** (upper/lower limits): `Data(censor=flags)` with `0/1/−1` per band (§3.3) — the existing `CensoredLikelihood` machinery ✓, now with a taught home ✓.

`sed.fit()` is an *un-deprecation* — after the Bagpipes ergonomics review, the one-liner won; it re-blesses an existing method as sugar while `ForwardModel.fit` stays canonical.

### 6.2 Catalog — homogeneous (same filters): the common case

```python
fwd = ForwardModel.build(sed=sed, observation=obs,
          approx=WavePrecomp(catalog_z_range=(0.05, 1.5), n_z=200))

cat  = Catalog(fwd, table, redshift_col="z", flux_unit="mJy")     # ✓ one noun
post = cat.fit(method="mcmc_nuts", key=key,
               forward_chunk_size=64)                              # K vmapped per lax.map step ✓
post["stellar_mass"]                                               # (N_galaxies,)

mock = cat.predict(param_table, chunk_size=4096)                   # ✓ → (N, n_filters)
Catalog.from_histories(...).simulate(lines=...)                    # ✓ sim SFH/Z tables, §8.1
# future: cat.simulate(noise=..., key=...)                         # ◆ #1312 — noisy draws, SBI
#                                                                  #   (refused today, not ignored)
```

**Column matching ✓: by name, by default.** Filters have registry names, so table columns `sdss_r` / `sdss_r_err` match automatically; `flux_cols=`/`err_cols=` remain as explicit positional overrides (validated by count). The swapped-column silent failure dies. Censor flags via `censor_cols=`.

**Units ✓: `flux_unit=` required for table-in** (`"mJy"`, `"cgs_fnu"`, `"maggies"`, `"ab_mag"`) — no default, no guessing; converters exist (`conversions.py`). Arrays-in stays documented cgs.

**NaN policy ✓: error by default**, naming rows/bands and counts; the message teaches `missing="mask"` (NaN → absent band, §3.3). Sentinels (`−99`) are never auto-interpreted.

**Known redshifts are first-class ✓.** Redshift is a *column*; `catalog_z_range` makes the whole catalog **one compile** (`sed_model.py:188-212`) — each row's z flows in as a runtime ztable interpolation (~µs). `Catalog` sets the per-row Fixed z internally and validates the span at construction ✓ — no `Fixed(0.0)  # placeholder` idiom in taught examples. Free-z per galaxy: supported iff the model has `redshift=Distribution(...)` and no `redshift_col`; both given → error ✓. Photo-z uncertainties: future, [#1314](https://github.com/suchethac/tengri/issues/1314).

**Shapes.** Table-in or arrays-in; **internally always materialized contiguous**: flux/noise `(N, n_data)`, redshift `(N,)` (§9.1). Spectra catalogs are arrays-in `(N, n_pix)` on the shared instrument grid (table-in is n/a for spectra) ◆.

**Scaling knobs ✓.** `forward_chunk_size=K` (XLA graph O(1) in N); `n_pad="auto"` (shape-bucket catalog sizes to reuse one compile).

**The `fit_batch` cliff (W2, [#1316](https://github.com/suchethac/tengri/issues/1316)).** Today `fit_batch(redshift_col=…)` clones a fresh `SEDModel` per row → new signature → full recompile per galaxy, silently. `Catalog` must auto-enable/validate `catalog_z_range`; `fit_batch` is now a deprecated alias of `Catalog.fit` ✓. Wave 0 shipped the loud warning (#1326); the zero-clone half was blocked on [#1329](https://github.com/suchethac/tengri/issues/1329) — `SEDModel.fit` had no `params=` parameter and validated no unknown kwargs, so the per-row override the WavePrecomp docstring advertised (`model.fit(row.data, params={"redshift": row.z})`) was silently swallowed. Both have since shipped: the `params=` plumbing plus unknown-kwarg validation, and the runtime-redshift sequential path.

### 6.3 Catalog — heterogeneous (different filters per galaxy)

Contract: **a galaxy filter not in the LUT set → raise.** ([#1317](https://github.com/suchethac/tengri/issues/1317))

```python
# fallback that works today — sequential, per-galaxy compile          ✓
Catalog(fwd, galaxies).fit(method="map", forward_chunk_size=1)
#   forward_chunk_size=1 → no vmap → ragged n_data allowed

# target — vmapped via union-LUT + presence mask                      ✓
approx = WavePrecomp(catalog_z_range=(0.05, 1.5))   # filters come from the Observation
fwd    = ForwardModel.build(sed=sed,
             observation=Observation(photometry=Photometry.from_names(union_filter_set)),
             approx=approx)
cat    = Catalog(fwd, table, redshift_col="z", flux_unit="mJy")
post   = cat.fit(method="mcmc_nuts", key=key, forward_chunk_size=64)
```

**Shapes (union path) ✓.** Rectangular over the union: `flux`, `noise` `(N, n_union)`, plus **two distinct channels** (§3.3): `presence (N, n_union)` bool — `False` = band absent, χ² skips it — and `censor (N, n_union)` in `{0,1,−1}` for limits in bands that *were* observed. The model predicts all `n_union` bands from the single union LUT; the masks select per galaxy. A band a galaxy has but the union lacks → **raise**. (The presence mask is a **new channel**, not a reuse of `data_mask` — that name already means censoring, with boolean arrays rejected by design.)

### 6.4 Hierarchical — one joint model; data at fit (W4, [#1319](https://github.com/suchethac/tengri/issues/1319))

> **Two-track decision (2026-08-05, decisions log 23):** this section is Track A — the binding standardization target. At realistic N the *production* fit is Track B: two-step estimation — MAP/Laplace-fit each galaxy, then fit the population over the interim posteriors ([PR #1479](https://github.com/suchethac/tengri/pull/1479); status in `docs/dev/hierarchical-psd-handoff.md`). The joint fit below is the small-N / validation form.

Your science case: N galaxies, each with per-galaxy parameters θᵢ (mass, dust, its SFH latent field ξᵢ), sharing **one** burstiness statistic — the PSD hyperparameters σ, τ take a single value for the whole population, and every galaxy's data pulls on it.

```python
fwd = ForwardModel.build(
    mode="hierarchical",                                    # ◆ assertable; inferred if omitted
    sed=template_sed, observation=obs,
    shared=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"),
    approx=WavePrecomp(catalog_z_range=(0.05, 1.5)))

post = fwd.fit(pop_data, method="vi", key=key)              # ◆ data at fit — N inferred here
post.shared_samples["sfh_field_psd_sigma"]                  # population hyperparameter
post.properties["stellar_mass"]                             # per-galaxy, (N,)
```

**Semantics.** `shared=(names,)` = **literal one-value sharing** — one value for all N; its prior is the template's prior on that parameter. Everything else stays per-galaxy. Partial pooling (θᵢ ~ N(μ,τ), fit μ,τ) is a *different* future feature, reserved as `pooled=` — never overloading `shared=`.

**Data at fit (adopted).** The uniform rule (§3) extends here: even a hierarchical ForwardModel holds no measured values. `pop_data` is N records of the shared schema (a `Data` batch / table; homogeneous-grid contract — same filters for all N). N is inferred at the action: from the data at `fit`, from the params' leading axis at a batched `predict`. The fully-materialized joint spec `{φ_shared, θ₁…θ_N}` exists once data (or an N) is supplied — honest, since it *is* data-dependent.

**Scaling contract ◆ (binding ≠ jitting).** Nothing about the API implies compiling an N-galaxy XLA graph:

```
per-galaxy forward kernel      ← JIT'd ONCE at single-galaxy scale
        │                         (shared via the structural kernel cache)
        ▼
jax.lax.map(kernel, …, batch_size=K)
        │                      ← a COMPILED, DIFFERENTIABLE LOOP:
        │                         graph size O(K); compile time O(1) in N
        ▼
Σ per-galaxy χ² + shared-param broadcast   ← thin, nearly free
```

The contract, binding on any implementation: **compile cost O(1) in N** (chunked `lax.map`, graph O(K)); per-galaxy kernels compiled once at galaxy scale and reused; the population-level combination is a thin differentiable layer (it must stay inside the trace — shared-parameter gradients accumulate across all galaxies — but `lax.map` gives autodiff with a loop, not an unrolled graph); **no O(N) Python in the hot path**. Data always flows as traced arguments, so refitting a same-N resample recompiles nothing. (At N ~ 10⁷ a *joint* fit with per-galaxy latents is ~10⁹ parameters — sampler-limited, not compile-limited; that regime is subsampling or amortized/SBI, [#1312](https://github.com/suchethac/tengri/issues/1312).)

---

## 7. Defining the likelihood — a config, not a subclass

The likelihood never appears in the sampling loop the astronomer sees. The **`NoiseModel` lives on the `Observation`** (`observation.py:189` ✓) — it is the instrument's noise character, and it is compile-relevant (a Student-t swap changes the likelihood, hence the engine key):

```python
obs = Observation(
    photometry = Photometry.from_names([...]),
    noise      = NoiseModel(),                                    # diagonal Gaussian (default)   ✓
    #            NoiseModel(calibration_floor=0.05)               # fixed cal floor in quadrature ✓
    #            NoiseModel(calibration_floor=Uniform(0.01, 0.15))# cal floor as a free param     ✓
    #            NoiseModel(student_t_dof=10)                     # heavy-tailed, outlier-robust  ✓
)
# spectroscopy: marginalize a polynomial flux calibration (fit-time flag)          ✓
fwd.fit(data, calibration_marginalize=True, cal_n_poly=3, ...)
```

`sigma_eff = sqrt(sigma_obs² + (f_cal · model)²)`, optionally Student-t; censored bands via `Data.censor` (§3.3) feed `CensoredLikelihood` ✓. Escape hatch — the `Likelihood` protocol (`protocols/likelihood.py`): `log_prob(prediction, data, noise_params) -> scalar`, `declared_parameters()`, `name`; supplied as `fwd.fit(..., likelihood=MyLikelihood())` ✓.

The binding is CompoSED's `Problem`, minus one object:

```
CompoSED:  Problem(backend, parameters, data, likelihood, filters);  fit(problem, sampler)
tengri:    ForwardModel(sed + observation[schema + NoiseModel] + priors-in-spec)
           fwd.fit(Data, method=...)          └────── "the problem" ──────┘
```

**Noisy mock draws for SBI (future, [#1312](https://github.com/suchethac/tengri/issues/1312)).** `fwd.simulate(params, key=…)` / `Catalog.simulate(...)` draw noisy observations using the **same** `sigma_eff` the likelihood uses — the simulate/fit loop closes on one noise definition. The same verb serves mock catalogs whose histories come from a simulation rather than from parameters (§8.1, [#1396](https://github.com/suchethac/tengri/issues/1396)); `noise=` is what separates a noiseless prediction from a draw.

---

## 8. Prediction — one contract on both surfaces

`SEDModel.predict()` and `ForwardModel.predict()` return the **same** `Prediction` with the same accessors (`forward_model.py:404` ✓); moving between surfaces changes only which instrument-dependent accessors exist.

```python
pred = sed.predict(params)         # rest-frame physics; no instrument needed      ✓
pred.rest_sed(); pred.rest_sed(wave); pred.properties["stellar_mass"]

pred = fwd.predict(params)         # + instrument accessors                        ✓
pred.photometry(); pred.spectrum(wave_obs); pred.obs_sed()

fwd.predict_properties(params, names=("stellar_mass", "sfr"))   # the ONE jit/vmap surface ✓
```

Preserved rules (NAMING_CONTRACT §4b): `predict()` takes `params` and nothing else; arrays don't carry their axis (`pred.wave_rest`/`pred.wave_obs`); `obs_sed()` is L_ν (a *frame*, not a flux); bare `pred.rest_sed` raises.

**Shape contract for non-scalar scenes ◆:** accessors return the scene's natural shape — multi-population: summed by default, per-component via the existing `predict_*_components` surface (`forward_model.py:234` ✓); hierarchical: `(N, …)` with N from the params' leading axis; IFU: cube-shaped (detailed spec deferred with the spatial work). **Multi-population namespacing ◆:** parameters, priors, and `post.properties` keys use the dotted `"{pop}.{param}"` prefix, consistent with the `agn.L_bolometric` derived-state convention.

### 8.1 Simulation catalogs — SFH/Z histories in, photometry + lines out ✓ ([#1396](https://github.com/suchethac/tengri/issues/1396))

The other direction of `simulate`: you already know each galaxy's star formation
history, because a hydrodynamic simulation, a semi-analytic model, or an empirical
model like UniverseMachine produced it. You want observed photometry and emission
lines for N of them, fast.

```python
sed = SEDModel.build(
    ssp_data=ssp, observation=obs,
    sfh={'type': 'table'},          # histories arrive at the action, not at build
    met={'type': 'table'},
    dust={'type': 'calzetti'}, neb={'type': 'cue'},
)
fwd = ForwardModel.build(
    sed=sed,
    approx=(WavePrecomp(catalog_z_range=(0.0, 3.0)), FeaturePrecomp()),
)

cat = Catalog.from_histories(                    # ✓
    fwd,
    t_gyr    = t_gyr,                # (n_t,) shared grid, or (N, n_t)  [Gyr, cosmic time]
    sfr      = sfr,                  # (N, n_t)                         [Msun/yr]
    met      = logzsol,              # (N, n_t)                         log10(Z/Zsun)
    redshift = z,                    # (N,)
    params   = {"dust_tau_v": tau},  # (N,) per-galaxy scalars
)

mock = cat.simulate(lines=("Halpha", "OIII_5007"), chunk_size=4096)   # ✓

mock.photometry                  # (N, n_filters)  [erg/s/cm²/Hz]
mock.lines["Halpha"]             # (N,)            [erg/s/cm²]
mock.properties["stellar_mass"]  # (N,)
mock.to_table()                  # flat column dict, parquet/FITS-ready — the §9.3 table-OUT leg

noisy = cat.simulate(..., noise=obs.noise, key=key)   # ◆ #1312 — raises today, does not
                                                      #   silently return a noiseless mock
```

`from_histories` is a classmethod, not a new noun — the result **is** a `Catalog`, so
`.fit()` on it stays meaningful (fit dust or redshift at a known, fixed SFH: a
genuinely useful validation mode, and the cleanest way to ask what a survey could have
recovered from a simulated galaxy).

**Histories are records, not parameters.** This falls out of §3's uniform rule rather
than extending it: a simulation's SFH is a per-galaxy record, exactly like a flux
vector, so it enters at the action. The physics is already built this way — the `table`
SFH declares **zero** free parameters ("the table IS the SFH"), and the arrays arrive
at runtime as `params["sfh_t_gyr"]`, `params["sfh_sfr"]`, `params["met_history"]`.

**Why the LUT still applies — the whole speed argument.** `WavePrecomp` bakes the
SSP × filter integral (§5), and that integral is **SFH-independent**: a tabulated SFH
changes only the (met, age) weights the SSPs are summed with, never the per-SSP band
fluxes. So the expensive precompute is built once and reused unchanged for every
galaxy in the simulation; per-galaxy cost collapses to a weight computation and a
matrix product. `FeaturePrecomp` does the same for lines. With `catalog_z_range`
keeping per-galaxy redshift a runtime ztable lookup, the whole catalog is **one
compile** — the §9.4 compile-reuse contract, applied to histories instead of fluxes.

**Shapes.** `(N, n_t)` history channels materialize contiguously alongside `(N,)`
scalar parameters (§9.1). Fixed `n_t` ⇒ one compile signature; ragged histories are
padded or shape-bucketed via `n_pad`, exactly as ragged catalogs are.

**Shipped.** The four gaps this section used to list are closed
([#1396](https://github.com/suchethac/tengri/issues/1396)). For the record, they were:
`Catalog.predict` read only `spec.free_params`, so the history arrays were never seen;
its `np.stack(…, axis=1)` could not mix `(N,)` scalars with `(N, n_t)` histories; it
returned photometry only, with no line channel; and the `FeaturePrecomp` weight path
silently returned a **zero** SFH for `sfh={'type':'table'}`
([#1395](https://github.com/suchethac/tengri/issues/1395)) — photometry still looked
correct, which made the zero lines and zero stellar mass harder to spot. All four are
replaced by one uniform column channel (`name → array` with any trailing shape) plus a
guard that now raises. The one-compile claim is asserted by a test that measures
`jit(vmap(...))` against bare `vmap` (236 dispatches vs 1) and fails if the `jit` is
removed.

**Closed since this section was written** ([#1538](https://github.com/suchethac/tengri/pull/1538),
on `main` as `547cee357`). Both are recorded rather than deleted, because the second
explains how the first shipped green:

- **A tabulated SFH silently dropped mass older than the SSP grid's oldest age bin**
  ([#1522](https://github.com/suchethac/tengri/issues/1522), closed) — the one that
  mattered most here, because a simulation history starts at t≈0 by construction. With
  a PARSEC/MILES SSP (oldest bin 12.589 Gyr) a constant SFH lost 8.8% of its mass at
  z=0 and an exponentially-declining one lost **46%**; the photometry carried the same
  factor. The CIC integrand now extends past the oldest template, so that mass lands
  on it instead of falling off the end: `stellar_mass / table integral` = 0.99998244 on
  a grid short enough to truncate, *identical* to a grid where nothing falls off. What
  remains is a **color** approximation — those stars wear the oldest template's
  spectrum, no older one existing — announced by `SFHBeyondSSPGridWarning` rather than
  left silent. `age_kernel='dsps'` still truncates, its histogram kernel having no bin
  past the last template, and says so.
- **The parametric round-trip test is now on `main`** — #1396's acceptance list called
  for "a tabulated history that reproduces a parametric SFH gives photometry matching
  the parametric model's — the test that proves the histories are actually being used",
  and it was the one criterion never written. Every test shipped before it compared
  table against table, on a *synthetic* SSP, with ratio-only assertions. Truncation is
  a common factor and a ratio divides it out, which is exactly how the defect above
  passed 33 of them. Measured agreement between the arms: 1.9e-4 in color, 1.8e-5 in
  formed mass, with the resolving power pinned by a negative control rather than
  assumed (τ×2 moves the color by 9.2e-2, 92× the tolerance).

  Writing it surfaced a second blind spot in the fixture, not the code: `synthetic_ssp_wide`
  is **separable** — `base(wave) * f(age) * g(met)` — so its wavelength shape is identical
  at every age and the SED shape cannot respond to the SFH at all. Doubling τ moves its
  colors by 3.6e-14. Any color assertion written against that fixture is unfalsifiable,
  which is a trap for more than this section.

**Residuals — what is still not true.** Measured on `main`, 2026-08-05:

1. `simulate(lines=…)` accepts only the five `DESI_LINES` names, and the error advising
   "pass `LineDef` objects directly" raises that same error when you do — `by_name` is
   keyed by string. [OII]λ3727 and Lyα are unreachable.
2. `simulate()` hardcodes `fast=True` with no `fast=` parameter, so a model outside the
   fast path's supported chain cannot simulate lines at all — not even slowly. Refusing
   to degrade silently is right; having no escape hatch is not.
3. `_map_chunks` crashes with bare `ZeroDivisionError` / `IndexError` on `N=0` (an
   ordinary empty selection cut) and on `chunk_size ≤ 0`, below the domain-error
   standard the rest of the file holds to.
4. #1396 records that `compute_joint_weights` supports **delta metallicity only**, so
   the `met={'type':'table'}` in the example above falls back to the exact forward —
   correct, but it forfeits the fast line path that motivates the example.

The last of the four original gaps — the silent zero — is **closed twice over**.
`compute_joint_weights` now refuses a
tabulated SFH with a reason rather than laundering a zero, which made the failure loud
— and the window-LUT line path then went further and *serves* tabulated histories
outright. Measured on `main` 2026-08-03, `measure_line_fluxes(params, defs, fast=True)`
against a `sfh={'type':'table'}` model agrees with the exact forward to **1 ULP**
(`max |fast/exact − 1| = 2.2e-16`) on Hα and [OIII]λ5007. `_require_feature_fast_eligible`
gates on the *component chain*, not the SFH type, so a table is eligible wherever a
parametric SFH is.

One asymmetry survives, and it is why `Catalog._prediction_columns` exists: the fast
path reads fixed scalars (`met_logzsol`, `dust_tau_bc`, …) straight out of `params` and
raises `KeyError` on a free-params-only dict that `fast=False` accepts happily. The
catalog merges `spec.get_fixed_values()` so every channel sees a complete dict; a
direct caller has to do it themselves, and nothing says so.

---

## 9. Arrays, memory, and results

### 9.1 Input — always materialize contiguous (adopted)

Catalog surfaces accept **table-in** (columns, name-matched) or **arrays-in** (power user), but **internally always materialize contiguous arrays** before any JAX transform: flux/noise `(N, n_data)`, redshift `(N,)`, presence/censor `(N, n_union)`. That is the shape vmap wants. `list[dict]` (today's input ✓) is accepted but immediately stacked — at N=10⁵ it otherwise carries ~10× memory overhead and N× host-to-device transfers.

> **Method note (#1363).** These snippets deliberately use `mcmc_nuts` (or `map`).
> `native_vi_linear` is registered `tier="broken"` — `[UNSTABLE] … segfaults on
> DPL/dense_basis photometry mocks (#231)` — so `run()` refuses it without
> `allow_unvalidated=True`, *and* the batched native-VI path raises
> `NotImplementedError` for per-galaxy redshift and for presence masks, which are
> the exact features §6.2 and §6.3 exist to demonstrate. Do not restore it here
> until the registry tier says otherwise.

### 9.2 Output — flexible summary, not fixed quantiles (adopted; [#1313](https://github.com/suchethac/tengri/issues/1313))

For large N the sample cube `(N, n_samples, n_params)` (~8 GB at N=10⁵) must never be forced into memory. `CatalogPosterior` gains streaming, **configurable** summaries — arbitrary percentiles + arbitrary reducers, chunk-reduced at the `forward_chunk_size` boundary:

```python
post = cat.fit(method="mcmc_nuts", key=key, forward_chunk_size=64,
               store="summary",                              # vs "full"; default by N — switch is LOGGED
               percentiles=(2.5, 16, 50, 84, 97.5),
               reducers={"mean": jnp.mean, "std": jnp.std})
post.percentiles["stellar_mass"]      # (N, 5)
post.summary["mean"]["stellar_mass"]  # (N,)
post["stellar_mass"]                  # median convenience, (N,)
```

### 9.3 Results — one `Posterior` contract ✓

```python
post.properties["stellar_mass"]       # derived quantities (sugar: post.stellar_mass)  ✓
post.posterior_predictive(data, noise)  # predictive fluxes, residuals, chi² (dict)   ✓
post.summary(); post.save(path)       # ✓ (posterior.py:1945)
post.refine(...)                      # continue/refine a fit                          ✓
post.shared_samples[...]              # hierarchical only: population hyperparameters
cat_post.to_table()                   # ✓ table-OUT (parquet/FITS) — closes the CIGALE loop (#1317)
```

### 9.4 Caching — three tiers, and the surface-derived policy

| Tier | Depends on | Reusable across | Artifacts |
|---|---|---|---|
| **1 Physics** | model structure only | everything | `signal_response`, structural kernels, the SSP×filter LUT |
| **2 Problem shape** | + data **shape** (not values) | all same-shape galaxies | `loss_fn`, `grad_fn`, `logdensity_fn` |
| **3 This fit** | + data **values**, method | nothing | NUTS adaptation, mass matrix, MAP |

Tier 2 is galaxy-agnostic by construction — `data_args` is traced, never closed over (`backends/mcmc/_shared.py:25`) — the mechanism behind both `Catalog` vmap and the hierarchical scaling contract. All expensive caches are **model-keyed** (`_model_cache.py` WeakKeyDictionary), which is why `Fitter` has no state a fresh one lacks and stays internal.

**Compile-reuse contract (normative — binding on all implementation).** Compiled galaxy models are *always* reusable across different data:

- Data enters compiled programs exclusively as **traced arguments** — never closed over, never baked. Baking a per-galaxy value into a compile signature is a bug ([#1316](https://github.com/suchethac/tengri/issues/1316) is the canonical instance).
- Recompilation has exactly **four legitimate triggers**: model structure, data *shape* (bucketed via `n_pad`), free-parameter set, engine/method. Per-galaxy redshift is explicitly **not** a trigger (`catalog_z_range` ztable); per-galaxy data values are **never** a trigger.
- Minimize distinct compile signatures — prefer runtime inputs over baked constants wherever the numerics allow. LUTs (tier 1) are method-agnostic: one LUT serves MAP, HMC, and VI alike.
- Tier 3 is the only per-data artifact, and it must stay bounded (the surface-derived policy above).

**W3 ([#1318](https://github.com/suchethac/tengri/issues/1318)) ✓:** `forward.prewarm()` exposed (it was on the internal Fitter only). `lean=` retired — policy derives from the surface: `forward.fit()` = iterate policy (tier-3 kept, keyed by data fingerprint with cap 1 — re-running the same fit reuses it, a new galaxy replaces it, so loops never accumulate); `Catalog` = **also the iterate policy, deliberately** ([#1344](https://github.com/suchethac/tengri/issues/1344), resolved). The spec originally called for a distinct `Catalog` = sweep policy and that was **wrong**, which is why nothing in `src/` ever passed `'sweep'`: `_lean_keep_sig` is `compile_signature()` — data *shape*, never data values — so two galaxies of the same model and shape share one key. Measured: `keep_sig` and `_engine_cache_key()` are equal across two different galaxies. `iterate` therefore keeps the entry they share and the whole catalog pays **one** inference-body compile, while `sweep` (`keep_sig=None`) drops it and would recompile **per galaxy** — reintroducing the [#1316](https://github.com/suchethac/tengri/issues/1316) cliff the catalog path exists to remove. `sweep` stays reachable through `tengri.lean()` for memory-constrained runs willing to pay that recompile. `lean=True/False` survives as a hidden deprecated alias.

---

## 10. What peer codes taught us

- **Bagpipes** — nested-dict components + a one-liner fit → tengri's grammar + the re-blessed `sed.fit()`.
- **CIGALE** — catalog-native, table-in/table-out, redshift as a column → `Catalog` wholesale, including the table-OUT leg.
- **CompoSED** — likelihood as thin config; transparent chunked batching; never store the sample cube → `NoiseModel`, `forward_chunk_size`/`n_pad`, flexible summaries.
- **Prospector** — the anti-patterns: redshift buried in an obs dict; mandatory build-functions → redshift is a column/top-level key; construction is `build` classmethods + recipes.

---

## 11. Work items

Implementation ordering, absorbed-backlog mapping, and the near-term method focus (**MAP + HMC/NUTS first; VI off the critical path**) live in the epic: [#1322](https://github.com/suchethac/tengri/issues/1322).

| ID | Piece | Status | Tracks |
|---|---|---|---|
| — | single/catalog/hierarchical construct + predict/fit; one predict contract | ✓ | — |
| **W1** | `observation` optional on `ForwardModel.build` + inherit + LUT reuse-on-match + scoped guard | ✓ shipped | [#1315](https://github.com/suchethac/tengri/issues/1315) |
| **W1b** | IGM applied on every LUT path — **already true**; guard tests exist | ✓ (resolved stale) | [#1310](https://github.com/suchethac/tengri/issues/1310) |
| **W2 dep** | fit-time `params=` plumbing (per-row z injection; unknown-kwarg validation) | ✓ shipped | [#1329](https://github.com/suchethac/tengri/issues/1329) |
| **W2** | `Catalog` noun: table-in/out, name-matching, `flux_unit=`, NaN policy, union-LUT + presence mask, `to_table()`; `fit_batch` cliff + deprecation | ✓ shipped | [#1316](https://github.com/suchethac/tengri/issues/1316), [#1317](https://github.com/suchethac/tengri/issues/1317) |
| **W3** | `forward.prewarm()`; retire `lean` → surface-derived policy | ✓ shipped (Catalog sweep **withdrawn**, not pending — [#1344](https://github.com/suchethac/tengri/issues/1344) closed; see §W3 above) | [#1318](https://github.com/suchethac/tengri/issues/1318) |
| **W4** | hierarchical: `mode="hierarchical"` + `shared=`, data at fit, scaling contract; Population classes dissolve | ◆ two-track (decision 23): Track A standardization a must; Track B two-step estimator first instance merged ([PR #1479](https://github.com/suchethac/tengri/pull/1479)) | [#1319](https://github.com/suchethac/tengri/issues/1319) |
| **W5** | Observation razor + `Data` record + `mode=` validation | ✓ shipped | [#1321](https://github.com/suchethac/tengri/issues/1321) |
| **W6** | simulation catalogs: `Catalog.from_histories` + `simulate(lines=…)`, LUT-fast (§8.1) | ✓ shipped, including the parametric round-trip test whose absence had let [#1522](https://github.com/suchethac/tengri/issues/1522) through (see §8.1 for the four remaining residuals) | [#1396](https://github.com/suchethac/tengri/issues/1396), [#1395](https://github.com/suchethac/tengri/issues/1395), [#1522](https://github.com/suchethac/tengri/issues/1522) |
| — | per-axis `FeaturePrecomp.n_grid` | ✓ shipped | [#1311](https://github.com/suchethac/tengri/issues/1311) |
| — | flexibly-summarized `CatalogPosterior` | ✓ shipped | [#1313](https://github.com/suchethac/tengri/issues/1313) |
| — | `simulate` for SBI (the noise draw) | ◆ future | [#1312](https://github.com/suchethac/tengri/issues/1312) |
| — | photo-z uncertainties in catalogs | ◆ future | [#1314](https://github.com/suchethac/tengri/issues/1314) |

---

## 12. Decisions log

Chronological, with rationale — the *why*, not just the *what*.

1. **Public API unchanged; `ForwardModel.fit` canonical.**
2. **`SEDModel` keeps its (optional) observation** — enables the LUT and standalone prediction; a bare SEDModel predicts simulation SEDs.
3. **Authority: `ForwardModel.observation` wins** — measured: nothing reconciles the two today (silent-mismatch hole → W1).
4. **SEDModel = simple / ForwardModel = shape-general** — the boundary is output-shape complexity, not merely instrument presence.
5. **One predict contract** across surfaces.
6. **`Catalog` is one noun** with `.fit()`/`.predict()`/`.simulate()` — replaces the `CatalogFitter`/`CatalogPredict` pair (asymmetric, and "Fitter" is retired as a taught noun).
7. **Hierarchical is ONE `ForwardModel`** — the joint posterior does not factorize; `Catalog` has no `shared=`. (Corrected from the looser "catalog with shared params" framing.)
8. **`sed.fit()` kept as sugar** — the Bagpipes one-liner won; un-deprecation, not new API.
9. **`Fitter` internal** — all expensive caches are model-keyed; it has no state a fresh instance lacks.
10. **`lean` retired to a surface-derived policy; `forward.prewarm()` exposed.** Iterate policy: tier-3 keyed by data fingerprint, cap 1 (loops safe).
11. **Input always contiguous `(N, n_data)`; output flexibly summarized** (arbitrary percentiles + custom reducers).
12. **Observation razor (review round):** pure instrument schema — all filter/spectrograph flexibility (survey mixes, custom curves, bare wave arrays, LSF(λ), resolution matrices) already lives in `Photometry`/`Spectroscopy` and is untouched; only measured values move out.
13. **`Data` record adopted** — schema:record with Observation; validated at one seam; a Catalog table is N records of the schema; bare-array `fit(flux, err)` stays as sugar.
14. **Missing ≠ censored** — presence mask (catalog union only) and censor flags (`0/1/−1`, today's `data_mask`) are distinct channels, never one representation.
15. **Hierarchical data at fit** (user decision, overriding the data-at-build recommendation) — completes the uniform rule: *models never hold measured values; data enters at the action; N is inferred there*. Pre-fit, the joint spec is un-materialized — honest, since it is data-dependent.
16. **Binding ≠ jitting — the scaling contract:** compile O(1) in N (chunked differentiable `lax.map`, graph O(K)); per-galaxy kernels compiled once; no O(N) Python in the hot path. Holds regardless of API-level data placement.
17. **`mode=` inferred by default, assertable explicitly** — one builder; explicit users get one mode-aware validation error instead of scattered kwarg errors.
18. **`shared=` = literal sharing (tuple of names; prior = template's prior); `pooled=` reserved** for future partial pooling — the two are never overloaded onto one kwarg.
19. **Name-matched catalog columns by default; `flux_unit=` required for table-in; NaN → error that teaches `missing="mask"`.** Explicit over convenient; the error is the documentation.
20. **Deliverable = this spec; implementation = the tracked issues.**
21. **Simulation histories are records, not parameters** (§8.1) — an SFH/Z table from a hydro sim or SAM enters at the action via `Catalog.from_histories(...)`, never at model construction. This is the §3 uniform rule applied unchanged, not an extension of it: the `table` SFH already declares zero free parameters and takes its arrays at runtime. `from_histories` is a classmethod returning a `Catalog`, so `.fit()` on a simulated catalog keeps its meaning (fit dust or redshift at a known SFH).
22. **`simulate` is one verb with two sources.** Parameters or tabulated histories supply the SFH; `noise=` is the only thing separating a noiseless prediction from a draw. Splitting these into separate nouns would duplicate the chunking, the LUT wiring, and the table-OUT leg for no conceptual gain.
23. **W4 is two-track (user decision, 2026-08-05): standardization is a must; the realistic hierarchical fit is two-step.** Track A — the §6.4 construction (`mode="hierarchical"` + `shared=`, data at fit, scaling contract; Population classes dissolve) plus the flat backend seam ([#1394](https://github.com/suchethac/tengri/issues/1394) → [PR #1531](https://github.com/suchethac/tengri/pull/1531)) stays binding. Track B — at realistic N the production path is *two-step*: MAP/Laplace-fit each galaxy individually, then fit the population over the interim posteriors ([PR #1479](https://github.com/suchethac/tengri/pull/1479) merged: first instance, shared SFH-PSD (σ, τ); known per-galaxy-bias ceiling at N ≈ 32–64 — status and traps in `docs/dev/hierarchical-psd-handoff.md`). The §6.4 joint fit remains the small-N / validation form, not the production default. Neither track waits on the other.

---

## 13. Non-goals / explicit compatibility

- **No public API is removed.** `Fitter`, `CatalogFitter` (→ alias of `Catalog`), `PopulationFitter` (deprecated shim), `fit_batch` (→ alias) stay importable; `sed.fit` is un-deprecated, not added; `Observation(line_fluxes=…)` warns and forwards.
- **`ForwardModel.fit` remains the canonical inference surface**; `sed.fit` is sugar over it.
- **`Fitter` stays internal** — the cache-reuse mechanism, never taught.
- **W4 is two-track** (decision 23, 2026-08-05) — no longer an undated deferral: the §6.4 standardization is a must; the realistic-N production path is the two-step estimator (per-galaxy MAP/Laplace → population fit, [PR #1479](https://github.com/suchethac/tengri/pull/1479)).
- **IFU/spatial detailed shapes are deferred** with the spatial extension work; §8's shape contract reserves the space.
- **The deep `InferenceContext`/backend decoupling (ADR-0010) is out of scope** — internal hygiene, no user-visible payoff, gates nothing here.
