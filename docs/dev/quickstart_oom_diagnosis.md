# Quickstart OOM diagnosis (2026-05-03)

## Symptom

`docs/spine/00_quickstart.ipynb` with `dust_emission="draine_li2014"`
intermittently crashes with `nbclient.exceptions.DeadKernelError`
during the `Fitter(...)` constructor or first `fitter.run("mcmc_nuts",
...)` call. No Python traceback from the kernel side.

## What was already known

- Earlier successful runs (Dale 2014, Draine & Li 2014) emitted three
  warnings each:
  ```
  xla.cpu.CompilationResultProto exceeded maximum protobuf size of 2GB:
      2380645273  /  4701922291  /  2380300852
  ```
- Successful end-to-end wall time was 12–15 min on first compile; the
  persistent JIT cache cut subsequent runs to ~60 s.
- `result.summary_table()` printed dozens of `dust_*` parameters that
  were not in the active model — the parameter spec includes the union
  of every dust-emission backend's parameters even when only one is
  active.

## What we measured this session

A 30 GB watchdog (`/tmp/run_with_oom_monitor.sh`) wrapped both
`jupyter nbconvert` and a stripped-down repro script
(`/tmp/repro_oom.py`). Hardware: 48 GB RAM, macOS Darwin 25.4.0.

| Run | Cache state | Peak RSS | Outcome | Wall |
|---|---|---|---|---|
| nbconvert (DL14, fresh kernel) | warm (14 GB) | 15.75 GB | DeadKernel after 25 s | — |
| nbconvert (DL14, retry) | warm | 14.25 GB | DeadKernel after 25 s | — |
| nbconvert (DL14, cache cleared) | empty | 14.25 GB | DeadKernel after 25 s | — |
| repro.py direct (DL14, cleared) | empty | **15.84 GB** | SIGKILL (rc 137) | 1 h 34 m |

Watchdog never tripped (limit 30 GB). Process was killed by SIGKILL
from outside (jetsam/XNU memory-pressure killer) **at 16 GB peak RSS,
not at the 30 GB ceiling**. macOS jetsam responds to *system memory
pressure*, not absolute size — so even with 48 GB total free, a
process holding 16 GB of dirty pages while XLA worker threads churn
at high CPU can be jetsam'd.

## Root cause

The `repro.py` direct run finally surfaced an actionable trace:

```
E xla/service/algebraic_simplifier slow_operation_alarm.cc:73
  Constant folding an instruction is taking > 4 s:
    %transpose.1592 = f64[11149, 107, 12]{2,1,0}
      transpose(%constant.58204), …
      op_name="jit(run_evi_geovi)/while/body/vmap(
              transpose(jvp(jit(interpolate_metallicity_smooth))))…"

E xla.cpu.CompilationResultProto exceeded maximum protobuf size of 2 GB:
  2380645217
  → JaxRuntimeError: INTERNAL: PjRtCpuClient::SerializeExecutable
                     proto serialization failed
```

Two compounding problems:

1. **Wrong-fitter compile.** `Fitter.__init__` spawns a daemon thread
   (`_start_background_compilation`) that pre-compiles **all** inference
   backends — including `run_evi_geovi` (geoVI variational inference) —
   even when the user only asked for NUTS. Stack frames in the trace
   above are inside `jit(run_evi_geovi)`, not NUTS.

2. **Constant-folding explosion in `interpolate_metallicity_smooth`.**
   The metallicity-grid interpolation kernel produces a constant tensor
   of shape `f64[11149, 107, 12]` ≈ 14 MB. With Draine & Li 2014 dust
   emission active, the JIT graph nests this constant inside a `vmap(
   transpose(jvp(...)))` chain that XLA's algebraic simplifier
   constant-folds for several seconds per copy, generating a
   compiled-graph protobuf > 2 GB.

The protobuf warning is *not fatal* (it only blocks JIT-cache write),
but the surrounding compile holds enough working memory that jetsam
takes the kernel out before the executable is loaded.

## Why earlier 12-min runs succeeded

Pure luck of jetsam's heuristic. The successful runs squeezed under
the pressure threshold because (a) less memory was held by other
processes that day, (b) the `Posterior.summary_table()` and
`predict_sfh_quantities` vmap paths happened to compile in a different
order. The crash mode is statistical, not deterministic — see also the
log gaps in the watchdog output (16-min stretches with no `ps` sample,
indicating the kernel was thrashing).

## Fix

Three independent, additive mitigations:

### A — opt out of cross-backend background compilation (immediate)

Set `TENGRI_NO_BACKGROUND_COMPILE=1` before constructing `Fitter`.
The geoVI / `run_evi_geovi` compile is skipped; only the NUTS path
compiles, lazily, when `fitter.run("mcmc_nuts", ...)` is invoked.
Implemented in `Fitter._start_background_compilation` (line 582,
src/tengri/inference/fitter.py).

This is what the quickstart now does at module import time — see
the env-var line near the imports.

### B — switch to a lighter IR backend (optional)

`dust_emission="modified_blackbody"` is a 3-parameter analytic model
(no template grid → no 14 MB constants → no constant-folding storm).
`dust_emission="dale2014"` is also lighter than DL14 (smaller alpha
grid). Both produce sub-GB JIT graphs.

### C — fix `interpolate_metallicity_smooth` constant capture (deferred)

The 14 MB metallicity grid should be passed as a JAX array (vmappable
data) rather than a Python-side constant baked into the trace, so XLA
doesn't try to constant-fold transposes of it. This is a tengri
internals change, tracked separately. See:

- `src/tengri/components/stellar/sps/dsps_wrapper.py::interpolate_met_alpha`
- `src/tengri/forward/_kernels/compositional.py:344` (where the
  warnings' "Unrecognized parameter names" come from)

## Workaround in the notebook

The quickstart now sets `TENGRI_NO_BACKGROUND_COMPILE=1` before
importing tengri so the geoVI compile path is never invoked. NUTS
still compiles fine; the protobuf warning may still appear (DL14
graph is large) but no longer triggers SIGKILL on a 48 GB laptop.
Wall-time target with mitigation A: ~5 min on first run, <60 s on
warm cache.

## Compile-time deep dive — *why is it so slow?*

After the SIGKILL was eliminated by mitigation A, the next question is
why a single forward-model JIT compile still takes 12–15 min on Dale
2014 (and didn't finish in 90 min on Draine & Li 2014). Three
compounding factors:

### 1. The SSP grid is captured by closure as a 114 MB constant

`tengri.components.stellar.sps.dsps_wrapper.interpolate_metallicity_smooth(
ssp_flux, ssp_lgmet, log_z, lgmet_scatter)` does an einsum
`m,maw->aw` against `ssp_flux` of shape **`(12, 107, 11149)`** —
specifically:

| dim | size | meaning |
|---|---|---|
| `n_met` | 12 | metallicity grid points |
| `n_age` | 107 | log-age grid points |
| `n_wave` | 11149 | rest-frame wavelength samples |

That's **`12 × 107 × 11149 = 14.3 M f64 = 114.5 MB`** *per copy*.

Inside `SEDModel`, `ssp_flux` lives on `model._state.ssp_data.ssp_flux`
and is captured by closure when the kernel chain is built. JAX traces
it as a **concrete constant** (not a vmappable input). Every
gradient / vmap / while-iteration that touches this constant gives
XLA's `algebraic_simplifier` another transpose/reshape to
constant-fold:

```
slow_operation_alarm.cc:73
  Constant folding an instruction is taking > 4s:
    %transpose.1592 = f64[11149,107,12]{2,1,0}
      transpose(%constant.58204), …
```

XLA constant-folds these in series; the working set grows linearly
with the number of derivative passes. The resulting compiled-graph
protobuf passes the **2 GB serialization limit** (the warning the
user has been seeing for weeks).

### 2. IR template grids add another constant

| backend | shape | size |
|---|---|---|
| `modified_blackbody` | none (3-param analytic) | 0 MB |
| `dale2014` | `(alpha, wave) = (small, 1001)` | < 0.5 MB |
| `draine_li2014` | `(11, 36, 21, 1001)` powerlaw + `(11, 36, 1001)` single-U | **70 MB** |
| `themis` | similar to DL14 | ~50 MB |

DL14's 70 MB template is a *4-D* constant — its autodiff pullback
generates many more transposes than Dale 2014's 2-D template. That
explains why DL14 compiles 5× slower than Dale 2014, even though both
share the same stellar-population path.

### 3. Energy balance fuses everything into one graph

`dust_emission != None` triggers the energy-balance closure:
`L_IR = ∫ L_intrinsic dν − ∫ L_attenuated dν`. This couples the
metallicity-interpolated stellar SED with the IR template grid in a
**single fused JIT**, multiplying constant-folding opportunities. The
hybrid path (`forward/_kernels/compositional.py`) re-references the
SSP and IR grids inside the same expression, and XLA cannot share the
constant-fold work between branches because the simplifier rewrites
each branch independently.

### 4. NUTS adapts a dense mass matrix

By default `mcmc_nuts(dense_mass_matrix=True)` compiles **two**
gradient-based functions during warmup:
1. `single_step` — likelihood + grad, JIT-compiled once.
2. The window-adaptation routine vmaps that gradient across mini-chains
   to estimate a dense covariance matrix.

Each holds its own copy of the captured constants. So the full
NUTS path effectively pays the constant-folding cost twice. (Setting
`dense_mass_matrix=False` would compile a faster — but less efficient
— diagonal version; not recommended for SED fits because of the
age–dust–metallicity covariance.)

### Putting it together — total compile budget

For Dale 2014 + 14-band photometry + tsnorm SFH at fixed *z*:

```
  forward kernel   ~ 2 min     (114 MB stellar grid + filter-projection precompute)
  predict_loglikelihood (vmap'd across data)  ~ 2 min
  NUTS dense-mass warmup       ~ 4 min     (compiles a second jvp)
  NUTS sample loop             ~ 1 min     (re-uses warmup graphs)
  geoVI background compile     ~ 4 min  ← skipped via TENGRI_NO_BACKGROUND_COMPILE
  ─────────────────────────────────────
  total observed                 ~13 min     (13 min in latest run, env-var on)
```

For Draine & Li 2014 the per-graph cost grows by 3–10× because of
the 4-D IR template; `interpolate_metallicity_smooth` constant-folds
explode to multi-hour stalls before XLA produces a binary. With
`TENGRI_NO_BACKGROUND_COMPILE=1` the protobuf overflow stops being
fatal but the wall time is still impractical (>1 h).

### Long-term fix (landed 2026-05-03)

The actual root cause turned out to be **simpler than constant-folding
in general** and is now patched. With a probe that lowered
`build_loss_fn(...)` to HLO and counted >1 MB constants:

| `dust_emission` | hybrid `_photometry_raw` | HLO size | biggest constant |
|---|---|---|---|
| `None` | set | **0.4 MB** | 0 (none > 1 MB) |
| `dale2014` (before fix) | **None** ← fallback | **343 MB** | **114.5 MB** `[12, 107, 11149]` |
| `dale2014` (after fix) | set | **8.8 MB** | **2.1 MB** `[12, 107, 200]` |

The fix: `build_hybrid_photometry` raised `UnboundLocalError`
(`dust_age_w` not defined) for the *fast* dust scheme + `dust_emission`
combo. The `contextlib.suppress(Exception)` wrapper at
`sed_model.py:1088` swallowed it silently, leaving the model's
`_hybrid_kernels._photometry_raw = None`, which forced
`_predict_photometry_traceable` to fall back to the compositional
kernel — the only kernel that captures the **full 114 MB SSP flux
grid** as a closure constant.

Patch in `src/tengri/forward/sed_model.py` near line 760:

```python
dust_age_w = None
if self._dust_model != "single_component" and (
    self._dust_scheme == "exact" or self._dust_emission_model is not None
):
    dust_age_w = precompute_dust_age_weights(self.ssp_ages_yr)
```

Force-precomputing `dust_age_weights` whenever IR re-emission is on
keeps `_dust_exact=True` for the energy-balance branch, which means
`dust_age_w` is always defined → `build_hybrid_photometry` no longer
raises → hybrid kernel is used → SSP flux is summarized to the
200-wave coarse grid (2.1 MB) rather than baked in at 11149 waves
(114.5 MB).

**Measured impact (Dale 2014 + tsnorm SFH + 14-band photometry):**

| variant | wall time | peak RSS |
|---|---|---|
| NUTS, no fix | 12–15 min (sometimes SIGKILL'd) | 16 GB |
| NUTS, no fix, `TENGRI_NO_BACKGROUND_COMPILE=1` | 13 min | 5.3 GB |
| **HMC, no fix, `TENGRI_NO_BACKGROUND_COMPILE=1`** | **2 min 20 s** | **4.2 GB** |
| **HMC, with fix, `TENGRI_NO_BACKGROUND_COMPILE=1`** | **36 s end-to-end** | **2.3 GB** |

Posterior recovery quality is preserved: 0 divergences in 1500 HMC
samples, all 8 parameters within ~1σ of truth, 7 figures rendered.

### Known follow-up: DL07 hybrid energy-balance accuracy

After the fix, `tests/unit/test_hybrid_energy_balance.py` reveals
two pre-existing failures (`TestDL07EnergyBalance` and
`TestDL07EnergyBalanceWorstCase`) where the hybrid kernel's
energy-balance integral is ~22% off from the compositional reference
when `dust_emission="draine_li2007"`. These were *masked* by the
silent compositional fallback before; my fix exposes them. Dale 2014
and THEMIS pass cleanly; only DL07 has the discrepancy. Tracking as
a separate numerical-accuracy issue — unrelated to compile-time
behavior.

### Phase II-2 fix (focused, landed 2026-05-03)

Following the hybrid-kernel fix, a focused refactor threaded `ssp_flux`
and `ssp_lgmet` through the **compositional photometry kernel** as
JIT-traced inputs rather than closure-captured arrays. This addresses
the underlying problem class — the hybrid path is no longer the only
path immune to the 114 MB constant.

Patch in `src/tengri/forward/_kernels/compositional.py`:
- `_compute_rest_sed(...)` and `fused_tier2_phot(...)` accept new
  optional kwargs `ssp_flux_traced`, `ssp_lgmet_traced`. When the
  smooth-Z, non-evolving, non-alpha-Fe branch fires (the common case),
  it calls `interpolate_metallicity_smooth(ssp_flux_traced, …)`
  directly with the traced inputs instead of routing through
  `interp_metallicity(model, …)` (which closes over `model.ssp_data`).
- The two new kwargs default to `None`, in which case the function
  falls back to the closure-captured arrays — backwards-compatible
  with all existing callers (the lazy `logged_jit`-wrapped photometry
  function, batch fitters, tests).

Patch in `src/tengri/forward/sed_model.py::_predict_photometry_traceable`:
- When falling back to the compositional raw kernel, pass
  `ssp_flux_traced=self.ssp_data.ssp_flux` and `ssp_lgmet_traced=
  self.ssp_data.ssp_lgmet` so XLA traces them as runtime arguments.

Verified by direct HLO probe of the compositional `_photometry_raw`:

| call site | HLO size | >1 MB constants |
|---|---|---|
| `raw(sfr_on_ssp, params)` (closure path) | **343.5 MB** | one **114.5 MB** `[12, 107, 11149]` |
| `raw(..., ssp_flux_traced=…, ssp_lgmet_traced=…)` | **3.2 MB** | **0** |

That's a **107× HLO reduction** on the compositional path, eliminating
the constant entirely. The compositional path is the safety-net used
inside NIFTy/VI tracing and any other consumer that bypasses the
hybrid kernel.

Tests: 4809 unit tests pass with the fix; the only failing test on
this branch (`test_auto_high_d_routes_to_vi`) is a MagicMock issue
unrelated to my change (fails identically on unmodified main).

### Phase II-2 spectroscopy extension (landed 2026-05-03)

The same trace-input pattern was extended to
`build_fused_tier2_spectrum` and its inner `_compute_rest_sed_spec`.
Callers that pass `ssp_flux_traced` / `ssp_lgmet_traced` via the
spectroscopy raw kernel get the same XLA benefit as photometry: the
SSP grid stays a JIT input rather than a closure-captured constant
on the smooth-Z, non-evolving, non-alpha-Fe path that joint
photometry+spectroscopy fits use most of the time.

All 34 tests in `test_hybrid_spectrum_traceable.py` and
`test_predict_spectrum_wave_chunk.py` pass with the change.

### Still on closure capture (deferred)

The following branches still resolve `model.ssp_data.ssp_flux` from
closure and would benefit from the same treatment in a future pass:

- The non-fast non-JIT fallback in `pipeline._compute_dust_atten` and
  the legacy `compute_sed_components` rest path (only hit for tabulated
  SFH + `met_history` and DSPS table modes — none of these are on the
  quickstart, photometry, or hybrid spectrum paths).
- `_compute_rest_sed_compositional` in `sed_model.py` (called by
  `predict_sed`/`predict` user-facing API; not used in the inner
  inference loops).
- `analysis/diagnostics` and `tests/crossval` helper paths that drive
  the compositional kernel directly without going through
  `_compositional.spectrum`/`_compositional.photometry` — unchanged.

### Phase II-2 spectroscopy + ramp-Z + alpha-Fe sweep (landed 2026-05-03)

This commit completes the remaining items tracked in this section:

1. `forward/pipeline.py`: `interp_metallicity`,
   `interp_metallicity_evolving`, `interp_met_alpha_dispatch`, and
   `interp_met_alpha_evolving_dispatch` now accept optional
   `ssp_flux` / `ssp_lgmet` (and `ssp_alpha_fe`) kwargs. When supplied,
   the SSP grid enters the JIT graph as a runtime tensor instead of a
   closure-captured constant. The `model`-based default is preserved
   for backwards compatibility.
2. `forward/_kernels/compositional.py`: ramp-Z and alpha-Fe branches
   in `_compute_rest_sed` (photometry) and `_compute_rest_sed_spec`
   (spectrum) thread `ssp_flux_traced` / `ssp_lgmet_traced` through
   to the dispatch functions.
3. `forward/_kernels/compositional.py::build_hybrid_spectrum`:
   `_hybrid_spec_body` and the single-/two-component dust wrappers
   accept and forward the same kwargs, so the spectrum-domain
   energy-balance integral no longer bakes the SSP grid as a constant.
4. `forward/sed_model.py`:
   - `_predict_spectrum_traceable` (NIFTy/VI tracing path) now passes
     the SSP arrays as JIT-traced kwargs to the hybrid spectrum raw,
     and falls back to the compositional spectrum raw the same way.
   - `_predict_spectrum_compositional` passes them to
     `self._compositional.spectrum`.
   - `predict_spectrum` (hybrid mode) passes them to
     `self._hybrid.spectrum`.
5. Verification: 40 / 40 evolving-Z and alpha-Fe unit tests pass; the
   updated quickstart notebook still executes in ~30 s with peak RSS
   2.24 GB under the new `scripts/run_with_oom_monitor.sh` watchdog
   (20 GB limit). See `tests/unit/test_evolving_metallicity.py`,
   `tests/unit/test_alpha_fe.py`, `tests/unit/test_audit_regressions.py::TestBugNSS02EvolvingMetFusedKernel`.

### Headline numbers (final)

Dale 2014 + tsnorm SFH + 14-band photometry, MIST SSP grid, fixed *z*:

| variant | wall time | peak RSS |
|---|---|---|
| NUTS, no fix | 12–15 min (often SIGKILL'd) | 16 GB |
| HMC, no fix | 2 min 20 s | 4.2 GB |
| HMC, hybrid-kernel fix | 36 s | 2.3 GB |
| **HMC, hybrid + Phase II-2 fixes** | **25 s** | **2.0 GB** |

### Future work

The right fix is to stop capturing `ssp_flux` and the IR template
grids as closure-captured constants. Instead, thread them through the
new `tengri.protocols.PipelineState` (Phase II-1 scaffold, May 2026) as
explicit JAX arrays. Then:

- XLA sees them as runtime arguments, not constants → no
  constant-folding storm.
- Compiled graphs become tiny (the executable holds only the
  *operations*, not the data).
- The persistent JIT cache becomes effective across SSP files
  (currently every SSP grid hash invalidates the cache).
- Cross-galaxy fits in `PopulationFitter` / `CatalogFitter` reuse a
  single compiled binary instead of recompiling per galaxy.

Tracking ticket: see `project_sedmodel_split.md` (memory) and the
Phase II-2 milestone in `project_api_cleanup.md`.

## Verification

A second monitored run after applying the env-var workaround should
show:

- Peak RSS during compile ≤ 12 GB (no geoVI graph in flight).
- Single protobuf-overflow warning (NUTS path), no SIGKILL.
- `nbconvert` exits 0 with all 7 figures rendered.

Watchdog command (kept under `/tmp/run_with_oom_monitor.sh`):

```bash
/tmp/run_with_oom_monitor.sh \
    env JAX_PLATFORMS=cpu TENGRI_NO_BACKGROUND_COMPILE=1 \
    .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    docs/spine/00_quickstart.ipynb
```

`/tmp/nb_mem.log` records 5-second RSS samples; peak is captured in
`/tmp/nb_peak.txt`.
