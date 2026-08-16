# Wave 2 Catalog Implementation Plan (epic #1322 — #1317, #1318, #1313, fit_batch alias)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The astronomer-facing catalog surface: `Catalog(fwd, table, …)` with `.fit()` / `.predict()`, table-in/table-out, name-matched columns, explicit units, contiguous internals — defaulting to MAP/NUTS per the HMC/MAP focus.

**Architecture:** Six tasks. T1 builds the ingestion/validation core (table → contiguous arrays; pure, no inference). T2 wraps it in the `Catalog` class delegating to the existing engine. T3 adds `CatalogPosterior.to_table()` + flexible summaries (#1313). T4 exposes `forward.prewarm()` and retires `lean=` (#1318). T5 handles heterogeneous catalogs (union-LUT + presence mask). T6 turns `fit_batch` into a deprecated alias. T1→T2→T3/T5/T6 sequential where marked; T4 independent.

**Tech Stack:** Python 3.12, JAX, pytest + chex, ruff. Tables: whatever `fit_batch` accepts today (dict-of-columns / pandas / astropy — T1 Step 1 pins it).

**Depends on:** Wave 0 merged (esp. #1134 ztable defaults, #1316) and Wave 1 T1 (`Data`, for censor semantics). T4 depends only on Wave 0. **T2's per-row redshift injection additionally requires #1329** (fit-time `params=` plumbing + unknown-kwarg validation — the docstring-advertised override does not exist yet).

## Global Constraints

Identical to `docs/internal/plans/2026-07-23-wave0-api-fixes.md` §Global Constraints, plus:

- **Method focus (epic #1322):** `Catalog.fit()` defaults to `method="map"`; docs/docstrings recommend `"mcmc_nuts"` for posteriors. Never default to a VI backend; never gate on VI repair.
- **Compile-reuse contract (spec §9.4):** nothing in this wave may introduce a per-row compile trigger. Any new cache key must be justified against the four legitimate triggers in a code comment.
- The spec (`2026-07-23-inference-prediction-api-final.md` §6.2–§6.3, §9) is the API authority.

---

### Task T1: catalog ingestion core — table → contiguous, validated arrays

**Executor:** sonnet
**Branch:** `feat/catalog-ingest`

**Files:**
- Create: `src/tengri/inference/catalog_ingest.py`
- Test: `tests/unit/inference/test_catalog_ingest.py` (create)

**Interfaces:**
- Produces: `ingest_catalog(table, *, photometry, flux_unit, flux_cols=None, err_cols=None, redshift_col=None, censor_cols=None, missing="error") -> CatalogArrays` where `CatalogArrays` is a NamedTuple: `flux (N, n_bands) [cgs f_nu]`, `noise (N, n_bands)`, `redshift (N,) | None`, `censor (N, n_bands) | None`, `presence (N, n_bands) bool`, `n_galaxies int`, `band_names tuple`. T2 and T5 consume exactly this.

- [ ] **Step 1: Pin the accepted table types.** Paste: `rg -n "def fit_batch" -A 20 src/tengri/forward/convenience.py | rg -n "table|catalog|row|column"` and how rows/columns are accessed today. Support exactly: (a) a mapping of column-name → array, (b) any object with `__getitem__[col] -> array` and `len()` (covers pandas/astropy without importing either). No new dependencies.
- [ ] **Step 2: Failing tests** — the contract, one behavior each:

```python
# tests/unit/inference/test_catalog_ingest.py
"""#1317: table -> contiguous validated arrays. Name-matching by
default, explicit units, NaN policy that teaches, censor separate
from presence (spec 6.2-6.3, 9.1)."""
import numpy as np
import pytest


def _phot():
    from tengri.observation import Photometry
    return Photometry.from_names(["sdss_g", "sdss_r"])


def _table(**over):
    base = {"sdss_g": np.array([1.0, 2.0]), "sdss_g_err": np.array([0.1, 0.1]),
            "sdss_r": np.array([3.0, 4.0]), "sdss_r_err": np.array([0.2, 0.2]),
            "z": np.array([0.1, 0.5])}
    base.update(over)
    return base


def test_name_matching_default_and_shapes():
    from tengri.inference.catalog_ingest import ingest_catalog
    ca = ingest_catalog(_table(), photometry=_phot(), flux_unit="cgs_fnu",
                        redshift_col="z")
    assert ca.flux.shape == (2, 2) and ca.noise.shape == (2, 2)
    assert ca.redshift.shape == (2,) and ca.band_names == ("sdss_g", "sdss_r")
    assert bool(ca.presence.all())


def test_missing_named_column_error_lists_candidates():
    from tengri.inference.catalog_ingest import ingest_catalog
    t = _table(); del t["sdss_r_err"]
    with pytest.raises(ValueError, match="sdss_r_err"):
        ingest_catalog(t, photometry=_phot(), flux_unit="cgs_fnu")


def test_explicit_cols_override_validated_by_count():
    from tengri.inference.catalog_ingest import ingest_catalog
    with pytest.raises(ValueError, match=r"1.*2|2.*1"):
        ingest_catalog(_table(), photometry=_phot(), flux_unit="cgs_fnu",
                       flux_cols=["sdss_g"], err_cols=["sdss_g_err"])


def test_flux_unit_required_and_converted():
    from tengri.inference.catalog_ingest import ingest_catalog
    with pytest.raises(TypeError):
        ingest_catalog(_table(), photometry=_phot())          # no flux_unit
    mjy = ingest_catalog(_table(), photometry=_phot(), flux_unit="mJy")
    cgs = ingest_catalog(_table(), photometry=_phot(), flux_unit="cgs_fnu")
    np.testing.assert_allclose(mjy.flux, cgs.flux * 1e-26, rtol=1e-12)
    # 1 mJy = 1e-26 erg/s/cm^2/Hz


def test_nan_errors_by_default_and_teaches_missing_mask():
    from tengri.inference.catalog_ingest import ingest_catalog
    t = _table(sdss_r=np.array([3.0, np.nan]))
    with pytest.raises(ValueError, match=r"missing=.mask."):
        ingest_catalog(t, photometry=_phot(), flux_unit="cgs_fnu")
    ca = ingest_catalog(t, photometry=_phot(), flux_unit="cgs_fnu",
                        missing="mask")
    assert not ca.presence[1, 1] and ca.presence.sum() == 3


def test_censor_cols_parallel_channel():
    from tengri.inference.catalog_ingest import ingest_catalog
    t = _table(sdss_g_censor=np.array([0, 1]), sdss_r_censor=np.array([0, 0]))
    ca = ingest_catalog(t, photometry=_phot(), flux_unit="cgs_fnu",
                        censor_cols={"sdss_g": "sdss_g_censor",
                                     "sdss_r": "sdss_r_censor"})
    assert ca.censor.shape == (2, 2) and ca.censor[1, 0] == 1
    assert bool(ca.presence.all())     # censored != absent (spec 3.3)


def test_ab_mag_conversion_with_error_propagation():
    from tengri.inference.catalog_ingest import ingest_catalog
    t = {"sdss_g": np.array([20.0]), "sdss_g_err": np.array([0.1]),
         "sdss_r": np.array([21.0]), "sdss_r_err": np.array([0.2])}
    from tengri.observation import Photometry
    ca = ingest_catalog(t, photometry=Photometry.from_names(["sdss_g", "sdss_r"]),
                        flux_unit="ab_mag")
    fnu = 10 ** (-0.4 * (20.0 + 48.60))
    np.testing.assert_allclose(ca.flux[0, 0], fnu, rtol=1e-6)
    np.testing.assert_allclose(ca.noise[0, 0],
                               fnu * np.log(10) / 2.5 * 0.1, rtol=1e-6)
```

- [ ] **Step 3: Run — FAIL (module absent).** Paste.
- [ ] **Step 4: Implement `catalog_ingest.py`.** Pure NumPy at ingestion (host-side; JAX conversion happens in T2). Units: reuse `tengri.utils.conversions` where a converter exists (`rg -n "maggies|def .*_to_fnu" src/tengri/utils/conversions.py`); implement `"mJy"` (×1e-26), `"uJy"` (×1e-29), `"ab_mag"` (f_ν = 10^(−0.4(m+48.60)), σ_f = f·ln10/2.5·σ_m — cite Oke & Gunn 1983 in the docstring per citation rules), `"maggies"` (via existing converter), `"cgs_fnu"` (identity). Unknown unit → `ValueError` listing the valid set. `missing="error"` message must contain the literal string `missing="mask"`. Name-matching: for each `photometry.names` entry expect `f"{name}"`/`f"{name}_err"`; on a miss, the error lists the table's actual columns. NaN in an **error** column with finite flux → always an error (no mask escape — an unknown uncertainty is not an absent band). Sentinels: values ≤ −90 are NOT auto-interpreted; mention `-99` in the NaN error text as "convert sentinels yourself".
- [ ] **Step 5: Run — PASS.** Paste. Lint. Commit `feat(inference): catalog ingestion core — name-matched, unit-explicit, contiguous. Refs #1317`.

---

### Task T2: the `Catalog` class (homogeneous catalogs)

**Executor:** sonnet
**Branch:** `feat/catalog-class`
**Depends on:** T1.

**Files:**
- Create: `src/tengri/inference/catalog.py`
- Modify: `src/tengri/__init__.py` (export `Catalog`), `src/tengri/inference/catalog_fitter.py` (deprecation shim only, Step 6)
- Test: `tests/unit/inference/test_catalog_class.py` (create)

**Interfaces:**
- Consumes: `ingest_catalog` (T1); the existing `CatalogFitter` engine; `ForwardModel` (with `mode`, Wave 1 T4).
- Produces: `Catalog(fwd, table, *, flux_unit, redshift_col=None, flux_cols=None, err_cols=None, censor_cols=None, missing="error")` with `.fit(method="map", *, key, forward_chunk_size=1, n_pad=None, **kw) -> CatalogPosterior` and `.predict(param_table, *, chunk_size=1024) -> ndarray (N, n_bands)`.

- [ ] **Step 1: Failing tests:**

```python
# tests/unit/inference/test_catalog_class.py
"""#1317: one noun, action verbs. Wraps the existing engine; ingestion
and validation happen at construction (fail fast, before any compile)."""
import jax
import numpy as np
import pytest


def test_construction_validates_eagerly(fwd_3band, table_3band_bad_missing_col):
    from tengri import Catalog
    with pytest.raises(ValueError):      # missing err column found at __init__
        Catalog(fwd_3band, table_3band_bad_missing_col, flux_unit="cgs_fnu")


def test_fit_default_is_map_and_returns_catalog_posterior(fwd_3band, table_3band):
    from tengri import Catalog
    cat = Catalog(fwd_3band, table_3band, flux_unit="cgs_fnu", redshift_col="z")
    post = cat.fit(key=jax.random.PRNGKey(0))       # no method= -> "map"
    assert post.n_galaxies == 3
    assert np.asarray(post["stellar_mass"]).shape == (3,)


def test_redshift_span_validated_against_catalog_z_range(fwd_3band_zrange, table_z_outside):
    from tengri import Catalog
    with pytest.raises(ValueError, match="catalog_z_range"):
        Catalog(fwd_3band_zrange, table_z_outside, flux_unit="cgs_fnu",
                redshift_col="z")


def test_predict_mock_shapes(fwd_3band, param_table_3rows):
    from tengri import Catalog
    cat = Catalog(fwd_3band, None, flux_unit="cgs_fnu")   # prediction-only: no data table
    mock = cat.predict(param_table_3rows)
    assert mock.shape == (3, fwd_3band.observation.photometry.n_filters)


def test_no_vi_default_anywhere(fwd_3band, table_3band):
    import inspect
    from tengri import Catalog
    sig = inspect.signature(Catalog.fit)
    assert sig.parameters["method"].default == "map"
```

Fixtures (local to the test file): `fwd_3band` = synthetic-SSP SEDModel + 3-band obs + `ForwardModel.build`; `fwd_3band_zrange` adds `approx=WavePrecomp(catalog_z_range=(0.05, 1.5))`; `table_3band` = 3 rows via the T1 helper shape; `param_table_3rows` = mapping of free-param name → length-3 array (derive names from `fwd.spec.free_params` and fill with prior midpoints).

- [ ] **Step 2: Run — FAIL (no `Catalog`).** Paste.
- [ ] **Step 3: Implement `catalog.py`.** Constructor: run `ingest_catalog` immediately (fail fast); store `CatalogArrays`; if `redshift_col` given, require the model's z mechanism: `Fixed` redshift + `catalog_z_range` covering `[z.min(), z.max()]` (reuse the #1316 validation — locate it, share the helper, do not duplicate) OR free redshift with NO `redshift_col` (both → the spec §6.2 error). `.fit()`: assemble per-galaxy dicts only as an adapter at the engine boundary (`# adapter: engine still takes list-of-dicts; arrays are the source of truth (spec 9.1); engine-native arrays are T5's problem`) and delegate to the existing `CatalogFitter(model, galaxies, data_type).run(method, key=…, forward_chunk_size=…, n_pad=…)`. `.predict()`: chunked vmap of `fwd.predict_photometry` over the param table (`jax.lax.map` with `batch_size=chunk_size` over stacked param arrays), per-row redshift injected the same way fit does. Both verbs set the **sweep** cache policy flag internally (T4 interface — until T4 merges, pass today's `lean=True`).
- [ ] **Step 4: Run — PASS.** Paste. Sweep `-k "catalog and not slow"`.
- [ ] **Step 5: Docstrings** (Tier 1: full numpydoc on `Catalog`, `.fit`, `.predict`; units in brackets; `method="map"` default documented with `"mcmc_nuts"` recommended for posteriors; cite the spec).
- [ ] **Step 6: Deprecated alias.** In `catalog_fitter.py`, add module-level `__getattr__`-style shim or subclass so `CatalogFitter(model, galaxies)` still works but emits a one-shot `DeprecationWarning` pointing at `Catalog` (use the repo's deprecation idiom). Existing tests for CatalogFitter must stay green **without** modification (warnings are fine; `filterwarnings` strictness — check `pyproject.toml` for `-W error` on DeprecationWarning in tests and, if strict, add the targeted ignore to the shim's own tests only).
- [ ] **Step 7: Lint; commit** `feat(inference): Catalog — one noun, fit/predict verbs, MAP default. Refs #1317`.

---

### Task T3: table-out + flexible summaries (#1313)

**Executor:** sonnet
**Branch:** `feat/catalog-posterior-summaries`
**Depends on:** T2.

**Files:**
- Modify: `src/tengri/inference/catalog_fitter.py` (`CatalogPosterior`, ~lines 74-215) and `src/tengri/inference/catalog.py` (`fit` kwargs pass-through)
- Test: `tests/unit/inference/test_catalog_summaries.py` (create)

**Interfaces:**
- Produces: `Catalog.fit(..., store="full"|"summary", percentiles=(16, 50, 84), reducers=None)`; `CatalogPosterior.percentiles[name] -> (N, n_pct)`, `.summary[reducer_name][name] -> (N,)`, `.to_table() -> dict[str, np.ndarray]` (column mapping — the same duck-type T1 ingests, so round-trips).

- [ ] **Step 1: Failing tests:**

```python
"""#1313: flexible summaries — arbitrary percentiles + custom reducers,
chunk-reduced; store="full" keeps today's behavior; to_table closes the
CIGALE loop."""
import numpy as np
import jax.numpy as jnp
import pytest


def test_percentiles_and_reducers_shapes(catalog_3band):
    post = catalog_3band.fit(key=KEY, method="mcmc_nuts", n_warmup=50,
                             n_samples=50, store="summary",
                             percentiles=(2.5, 16, 50, 84, 97.5),
                             reducers={"mean": jnp.mean, "std": jnp.std})
    assert post.percentiles["stellar_mass"].shape == (3, 5)
    assert post.summary["mean"]["stellar_mass"].shape == (3,)
    assert np.asarray(post["stellar_mass"]).shape == (3,)   # median convenience


def test_store_summary_drops_the_cube(catalog_3band):
    post = catalog_3band.fit(key=KEY, method="mcmc_nuts", n_warmup=50,
                             n_samples=50, store="summary")
    with pytest.raises(AttributeError):
        post.samples                                          # cube not retained


def test_store_full_keeps_todays_behavior(catalog_3band):
    post = catalog_3band.fit(key=KEY, method="map", store="full")
    assert post.n_galaxies == 3


def test_to_table_round_trips_column_mapping(catalog_3band):
    post = catalog_3band.fit(key=KEY, method="map")
    t = post.to_table()
    assert set(t) >= {"stellar_mass"} and len(t["stellar_mass"]) == 3
```

Mark the NUTS-using tests `@pytest.mark.slow` if runtime exceeds ~30 s locally; keep n_warmup/n_samples tiny.

- [ ] **Step 2: Run — FAIL.** Paste. **Step 3: Implement.** Reduction happens per completed galaxy/chunk inside the existing result-assembly loop (locate where per-galaxy `Posterior` objects or stacked samples are collected — `rg -n "posteriors|append" src/tengri/inference/catalog_fitter.py | head`); with `store="summary"`, compute `np.percentile(samples, pcts, axis=0)` + each reducer per property, then drop the samples before appending. `store` default: `"full"` if `N <= 1000` else `"summary"` — and `log`/`warn` the automatic switch with the word "summary" (never silent). `to_table()`: medians for every property + `"{name}_p{pct}"` columns when percentiles stored.
- [ ] **Step 4: Run — PASS; sweep `-k catalog`.** Paste. Lint. Commit `feat(inference): flexible catalog summaries (percentiles+reducers) and to_table. Refs #1313`.

---

### Task T4: `forward.prewarm()` + retire `lean=` (#1318)

**Executor:** sonnet
**Branch:** `feat/prewarm-surface-policy`
**Depends on:** Wave 0 only. Independent of T1–T3 (T2 wires to it on merge).

**Files:**
- Modify: `src/tengri/forward/forward_model.py` (add `prewarm`), `src/tengri/inference/fitter.py:1624-1691` (`prewarm`), `:2068-2103` (lean block) and the `run()` signature
- Test: `tests/unit/inference/test_cache_policy.py` (create)

- [ ] **Step 1: Failing tests:**

```python
"""#1318: intent is expressed by the surface. forward.prewarm() builds
tiers 1-2; fit() keeps tier-3 keyed by data fingerprint (cap 1);
Catalog uses the sweep policy; `lean=` becomes a hidden deprecated alias."""
import jax
import pytest


def test_forward_prewarm_exists_and_is_idempotent(fwd_3band):
    fwd_3band.prewarm()
    fwd_3band.prewarm()          # second call must be a fast no-op


def test_lean_kwarg_warns_deprecation(fwd_3band, mock_flux):
    flux, err = mock_flux
    with pytest.warns(DeprecationWarning, match="lean"):
        fwd_3band.fit(flux, err, method="map", key=jax.random.PRNGKey(0),
                      lean=True)


def test_iterate_policy_same_fit_reuses_tier3(fwd_3band, mock_flux):
    # behavioral proxy: second identical fit must be much faster than the
    # first (tier-3 kept). Wall-clock ratio, generous threshold.
    import time
    flux, err = mock_flux
    k = jax.random.PRNGKey(0)
    t0 = time.perf_counter(); fwd_3band.fit(flux, err, method="map", key=k)
    t1 = time.perf_counter(); fwd_3band.fit(flux, err, method="map", key=k)
    t2 = time.perf_counter()
    assert (t2 - t1) < 0.5 * (t1 - t0)
```

- [ ] **Step 2: Run — FAIL** (`prewarm` missing on ForwardModel; no deprecation on `lean`). Paste.
- [ ] **Step 3: Implement.** `ForwardModel.prewarm(self, *, data_shape=None)` — construct the internal `Fitter` the same way `fit` does (dummy data of the observation's shape when `data_shape is None`; zeros are fine — tier 1–2 artifacts are value-independent by the compile-reuse contract) and call its existing `prewarm()`. In `Fitter.run`: accept `lean=None`; when a caller passes a non-None value emit the one-shot `DeprecationWarning("lean= is retired: fit() keeps your warm caches (iterate policy); Catalog sweeps automatically. See #1318.")` and honor it for compatibility; when `None`, derive: `_cache_policy="sweep"` if an internal flag set by `Catalog` is present (add `run(..., _cache_policy=None)` private kwarg, underscore-prefixed, excluded from docs), else `"iterate"`. Iterate = today's smart-lean keep-matching behavior (it already keys on `compile_signature`, which fingerprints shape not values — verify and cite in a comment); sweep = today's `lean=True` drop-stale path. Net behavior change is minimal by design — this task renames the *policy seam*, it does not re-engineer eviction.
- [ ] **Step 4: Run — PASS.** Paste. **Step 5:** wire `Catalog` (if T2 merged) to pass `_cache_policy="sweep"`. Sweep `-k "cache or lean or prewarm"`. Lint. Commit `feat(inference): forward.prewarm(); retire lean= behind surface-derived cache policy. Refs #1318`.

---

### Task T5: heterogeneous catalogs — union filters + presence mask

**Executor:** sonnet — **orchestrator pairs on review; this is the riskiest task of the wave**
**Branch:** `feat/catalog-union-presence`
**Depends on:** T1, T2.

**Files:**
- Modify: `src/tengri/inference/catalog_ingest.py` (presence already produced), `src/tengri/inference/catalog.py`, and the χ² assembly point for the batched path (locate: `rg -n "chi2|residual|loss" src/tengri/inference/catalog_fitter.py src/tengri/inference/loss_functions.py | head -15`)
- Test: `tests/regression/test_issue_1317_union_presence.py` (create)

**Interfaces:**
- Consumes: `CatalogArrays.presence (N, n_union) bool` (T1).
- Produces: masked χ²: absent bands contribute exactly 0 to the likelihood and its gradient; a galaxy whose table carries a band not in the union observation → `ValueError` at ingestion.

- [ ] **Step 1: Failing tests:**

```python
"""#1317 stage (b): union-LUT + presence mask. Absent bands contribute
exactly zero to chi^2 AND its gradient; presence is not censoring."""
import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def test_absent_band_contributes_zero(fwd_union, table_two_surveys):
    from tengri import Catalog
    cat = Catalog(fwd_union, table_two_surveys, flux_unit="cgs_fnu",
                  missing="mask", redshift_col="z")
    # galaxy 0 has all bands; galaxy 1 lacks the last band (NaN -> masked)
    post_masked = cat.fit(key=jax.random.PRNGKey(0), method="map")
    # reference: same fit with galaxy 1's absent band REMOVED via a
    # smaller observation must give the same galaxy-1 MAP
    post_ref = _fit_galaxy1_without_last_band(...)  # build inline: 2-band obs + single fit
    np.testing.assert_allclose(post_masked.map_for(1), post_ref.map_params(),
                               rtol=1e-5)   # adapt accessors


def test_gradient_is_zero_wrt_absent_band(fwd_union, table_two_surveys):
    # differentiate the assembled loss w.r.t. the masked flux entry; must be 0.0 exactly
    ...


def test_band_outside_union_raises(fwd_union, table_with_extra_band):
    from tengri import Catalog
    with pytest.raises(ValueError, match="not in the observation"):
        Catalog(fwd_union, table_with_extra_band, flux_unit="cgs_fnu")
```

Complete the two elided bodies before running — the plan requires it: the reference fit builds a 2-band `Photometry.from_names` observation + a single `fwd.fit`; the gradient test wraps the located loss callable with `jax.grad` w.r.t. the data argument and indexes the masked entry. If the loss callable cannot be isolated for a gradient probe, STOP and report — that is a design gap for orchestrator review, not something to approximate.

- [ ] **Step 2: RED (paste), Step 3: implement.** Masking enters at exactly ONE point: where per-band residuals are squared/summed for the batched path — multiply by `presence` (float) there, and divide degrees-of-freedom-dependent normalizations accordingly if any exist (check for `log(2*pi*sigma^2)` terms: a masked band's constant term must also drop — otherwise model comparison shifts; search the loss for the constant and gate it with the same mask). The `WavePrecomp(filters=union)` LUT needs no change — the model predicts all `n_union` bands; the mask selects. Guard rails: `presence` all-True short-circuits to the exact existing code path (bit-identical — assert in a test by comparing against a no-mask fit).
- [ ] **Step 4: GREEN (paste); Step 5:** sweep `-k "catalog or union or presence"` + one full fast-tier run (`…pytest tests/ -q -n 2`). Lint. Commit `feat(inference): heterogeneous catalogs via union filters + presence-masked likelihood. Refs #1317`.

---

### Task T6: `fit_batch` → deprecated alias

**Executor:** haiku
**Branch:** `chore/fit-batch-alias`
**Depends on:** T2.

**Files:**
- Modify: `src/tengri/forward/convenience.py:393-538`
- Test: `tests/unit/forward/test_fit_batch_alias.py` (create)

- [ ] **Step 1: Failing test:** `fit_batch(model, catalog, flux_cols=…, err_cols=…, method="map")` emits one-shot `DeprecationWarning` matching `"Catalog"` AND still returns the same-shaped result list as before (run once with warning captured, assert result length == n rows).
- [ ] **Step 2: RED (paste). Step 3:** implement — `fit_batch` body becomes: warn once, wrap the bare `SEDModel` in `ForwardModel.build(sed=model)` (observation inherited per #1315), construct `Catalog(fwd, table, flux_cols=…, err_cols=…, flux_unit="cgs_fnu", redshift_col=…)`, call `.fit(method=…, key=…)`, and adapt the return to the legacy list shape. The `flux_unit="cgs_fnu"` hard-code preserves legacy semantics (fit_batch always took cgs); note it in the warning text ("Catalog requires explicit flux_unit; fit_batch assumed cgs").
- [ ] **Step 4: GREEN (paste);** existing `fit_batch` tests stay green unmodified. Lint. Commit `chore(forward): fit_batch delegates to Catalog with a one-shot deprecation. Refs #1317, #1316`.

---

## Review protocol additions (orchestrator)

T5 gets a physics-grade review (masked-likelihood constant terms; gradient-zero proof re-run by me). T2/T6: run `python tools/check_doc_examples.py` (new public names). After the wave: one end-to-end demo script exercised manually — build → `Catalog` → `.fit(method="mcmc_nuts")` on 5 synthetic galaxies → `.to_table()` — before announcing the wave done.

## Self-review record

- Spec coverage: §6.2 (T1+T2), §6.3 (T5), §9.1 (T1), §9.2 (T3), §9.4 policy (T4), fit_batch alias (T6). `censor_cols` (spec §6.2) in T1. `simulate` deliberately absent (future, #1312).
- Placeholder scan: two elided test bodies in T5 are flagged as must-complete-before-run with their construction described; all other steps carry code.
- Type consistency: `CatalogArrays` fields defined in T1 are consumed by those exact names in T2/T5; `store/percentiles/reducers` names match #1313's issue sketch and spec §9.2.
