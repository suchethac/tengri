# Phase 3d — kernel adapter deletion + `approx=` enum (temporary plan, delete after merge)

**Status:** Phase 3d-A (typed `WavePrecomp` opt-in) shipped in PR #135;
kernel deletion deferred — see
[`orchestrator_ssp_threading.md`](orchestrator_ssp_threading.md) for the
SSP-as-traced-kwarg prerequisite that must land first.
**Date:** 2026-05-20.
**Delete after the implementing PR merges.**

## Goals

Two coupled changes shipped together:

1. **Delete `src/tengri/forward/_kernels/` entirely** — the 7 adapter classes, `hybrid.py` (~3400 lines), `compositional.py` (~700 lines), and the `__init__.py` re-exports.
2. **Replace dict-typed `approx={"wave_precomp": True, "ztable": True}` with a typed `WavePrecomp(...)` opt-in object** — `approx=WavePrecomp()` opts into the wave-precompute method with default ztable sampling; `WavePrecomp(n_z=200, z_min=0.0, z_max=3.0)` lets the user dial in the ztable grid. The default `approx=None` means *no approximation* (full wave-grid integration); `"exact"` is not a value because exactness is the absence of an approximation, not a chosen one.

After this, the user surface has **`compile=` (string enum) + `approx=` (None | `WavePrecomp(...)`)** and **no call-time `mode=` kwarg**:

```python
from tengri import SEDModel, WavePrecomp

# Default — exact wave-grid integration
model = SEDModel.build(observation=obs, ..., compile="per_component")

# Opt into the wave_precomp approximation, default ztable sampling
model = SEDModel.build(observation=obs, ..., compile="per_component",
                              approx=WavePrecomp())

# Same path, custom ztable sampling
model = SEDModel.build(observation=obs, ..., compile="per_component",
                              approx=WavePrecomp(n_z=200, z_min=0.0, z_max=3.0))

flux = model.predict_photometry(params)    # no mode= kwarg
```

`WavePrecomp` is a frozen dataclass mirroring `Fixed(...)`, `Uniform(...)` — typed, autocomplete-friendly. The component-level boolean flags (`wave_precomp`, `ztable`) survive as private implementation; build-time logic resolves the typed object into the flags it implies and threads `n_z` / `z_min` / `z_max` into the stellar component's `redshift_spec`.

After this PR, the only forward paths are:
- `observation.predict()` — exact, wave-grid integration. Compositional is the default semantics.
- `observation.predict_via_precomp()` — used automatically by `predict_photometry` when the model was built with `approx=WavePrecomp(...)`.

`predict_via_precomp` doesn't grow spectrum support in this PR — spectrum stays exact via `observation.predict()`.

## `approx=` semantics

| Value | What happens | When you'd use it |
|---|---|---|
| `None` (default — kwarg omitted) | full wave-grid integration end-to-end; no precomputed SSP×filter integrals | reference / bit-exact runs; small grids; debugging |
| `WavePrecomp()` | precompute the SSP×filter integral on the wavelength grid once at build; per-call filter integral becomes a lookup. **Free redshift transparently uses a ztable interpolation** built on the same LUT — no separate user flag. Default ztable: `n_z=100`, bounds from the redshift prior with 1 % padding. | hot inference loops; large filter sets; production fitting |
| `WavePrecomp(n_z=…, z_min=…, z_max=…)` | same method, user-controlled ztable sampling | hand-tuning the ztable resolution / range |

**Free-redshift handling** (internal, not a user flag): when `approx=WavePrecomp(...)` and `redshift` is free, the stellar component publishes the ztable interpolation in addition to the wave-precomputed LUT. The ztable is a free-z extension of the wave_precomp method, not a separate approximation. `model.spec.summary()` shows: `approx: wave_precomp (ztable: enabled for free z)`.

**Future extensibility:** new approximations become new dataclasses (e.g. `DustTaylor(...)` for the Charlot & Fall first-moment Taylor expansion). Stacking two methods would be a separate decision (likely `approx=(WavePrecomp(), DustTaylor())` if that ever lands) — but the single-method case stays a single object.

**Why not a dict:** the dict form encouraged users to think of `wave_precomp` and `ztable` as two independent toggles — but `ztable` was always derived from `wave_precomp` + free-z. A typed opt-in dataclass collapses that into one decision while still letting the user override the ztable sampling.

## Migration policy: hard replace

The dict form `approx={"wave_precomp": True}` is deleted in this PR with **no deprecation shim**. Rationale: the API is pre-1.0, the dict form has only been in `main` since #115 (~weeks), and call-site sweep is bounded:

- `tests/unit/test_stellar_precomp_publish.py` — ~13 call sites (all rewriting to `approx=WavePrecomp()`).
- `src/tengri/presets/synthesizer.py:119` — `approx=True` (legacy bool form, not dict; also dies).
- 1 notebook comment block (`notebooks/05_fitting_photometry.py:192-197`) — rewrite the comment.
- ~15 docstring / inline-comment references — sweep to the new vocabulary.

After the PR:
- `SEDModel(approx={...})` → `TypeError` (dict form removed).
- `SEDModel(approx=True)` / `approx=False` → `TypeError` (bool form removed).
- `SEDModel(approx="wave_precomp")` / `"exact"` / any other string → `TypeError` (string form removed).
- `SEDModel(approx=None)` (or kwarg omitted) → exact wave-grid path.
- `SEDModel(approx=WavePrecomp(...))` → accepted; anything else raises with the legal-values list.

## Scope

### Code surface changes

| Method on SEDModel | Action |
|---|---|
| `predict_photometry(params, mode=, approx=)` | drop `mode=`/`approx=`; body becomes `self.observation.predict(self.predict_via_orchestrator(params), {**fixed, **params})["phot_fnu"]` |
| `predict_spectrum(params, wave_obs=, mode=, ..., wave_chunk_size=)` | drop `mode=`/`approx=`; keep `wave_obs`, `wave_chunk_size`; route through `observation.predict()` which integrates `state.sed_intrinsic` through the spec wave grid |
| `predict_magnitudes(params)` | unchanged signature; internally calls `predict_photometry` |
| `predict_luminosity(params)` | unchanged signature; reads `state.sed_intrinsic / L_SUN` from `predict_via_orchestrator(params)` (today goes through compositional kernel) |
| `predict_rest_sed(params)` | returns a `SEDResult(wave=state.wave, sed=state.sed_intrinsic)` built directly from `predict_via_orchestrator(params)` |
| `predict_obs_sed(params)` | returns observed-frame redshifted SED built from `state.sed_intrinsic + state.sed_observed`-equivalent — uses existing IGM/DLA machinery without the compositional kernel |
| `predict_line_fluxes(params, ...)` | unchanged — already orchestrator-native |
| `predict_emission_lines(params)` | unchanged — already orchestrator-native |
| `predict_spectral_indices(params, ...)` | drop `mode=`; compute from `predict_rest_sed(params)` |
| `predict_*_components()` | unchanged — already orchestrator-native |
| `predict_photometry_batch / predict_spectrum_batch` | unchanged — `jax.vmap(...)` over the simpler `predict_photometry / predict_spectrum` |
| `predict_observables(params)` / `predict_observables_jit(params)` | unchanged |
| `predict(params) -> Prediction` | unchanged — uses orchestrator |
| `precompute_ztable(...)` | **delete** — it was building the legacy hybrid ztable kernel. `wave_precomp` is the replacement. |

### Internal SEDModel attributes to remove

- `self._compositional_kernels`, `self._build_compositional_kernels`
- `self._hybrid_kernels`, `self._build_hybrid_kernels`, `self._hybrid`
- `self._kernel_strategy` field on the dataclass + `strategy=` __init__ kwarg
- `self._predict_*_compositional`, `self._predict_*_hybrid`, `self._predict_*_auto`, `self._predict_*_traceable` private methods
- `self._observe_spectrum_from_rest_sed_chunked` (replace its callers with direct `compute_spectrum` chunked call)
- `_PREDICTION_MODES` constant + the mode-validation logic in `predict_photometry` / `predict_spectrum`

### Inference path

`src/tengri/inference/fitter.py:_init_precomputation` currently calls `model._build_hybrid_kernels()`. Replace with no-op or remove the call entirely; downstream inference goes through `model.predict_observables(params)` or `model.predict_photometry(params)` which no longer uses kernels.

### Public-API re-exports

`src/tengri/__init__.py` exports `KernelStrategy` and `NoCompatibleKernelError`. Replace with **`deprecated_alias`** stubs that raise `ImportError` with a clear migration message ("KernelStrategy is removed; the precompute LUT path is opt-in via `model.observation.predict_via_precomp` or by setting `approx=WavePrecomp()` at build time").

Same for the engineering aliases `DEFAULT_KERNEL_STRATEGY`, `LOW_MEMORY_KERNEL_STRATEGY`, `COMPOSITIONAL_ONLY_KERNEL_STRATEGY`, `EXACT_ONLY_KERNEL_STRATEGY`.

### Notebooks

| Notebook | Change |
|---|---|
| `notebooks/01_why_jax.py` | 9 hardcoded `mode="compositional"` calls → drop `mode=` (default is now exact wave-grid). Add a separate cell demonstrating `model.observation.predict_via_precomp(state, params)` as the opt-in fast path. |
| `notebooks/00_quickstart.py`, `05_fitting_photometry.py`, `06_fitting_spectroscopy.py`, etc. | use default `predict_photometry(params)` / `predict_spectrum(params)` — no mode= |
| `notebooks/09_parameter_sweeps.py` | unchanged — already uses default |

### Benchmark scripts

| Script | Change |
|---|---|
| `bench/scripts/benchmark_forward_model.py` | rewrite to compare `predict_photometry` (exact) vs `predict_via_precomp` (LUT). Drop the 3-way mode comparison. |
| `bench/scripts/benchmark_population_native.py` | drop `mode=` plumbing |
| `bench/scripts/benchmark_joint_indices_e2e.py` | drop `mode=` plumbing |

### Tests to update or delete

| File | Action |
|---|---|
| `tests/unit/test_mode_comparison.py` | **delete** — entire purpose was comparing modes |
| `tests/unit/test_mode_kwarg_deprecation.py` | rewrite: assert `predict_photometry(mode=...)` raises `TypeError` (kwarg removed entirely) |
| `tests/unit/test_precompute_kernel_invariants.py` | **delete** — tests internal kernel invariants |
| `tests/unit/test_hybrid_spectrum_traced.py` | **delete** — tests deleted hybrid spectrum kernel |
| `tests/unit/test_cue_hybrid_diagnostic.py` | rewrite without the hybrid path (Cue exact vs Cue via predict_via_precomp) |
| `tests/unit/test_hybrid_energy_balance.py` | **delete** — hybrid-specific |
| `tests/unit/test_fused_kernels.py` | **delete** |
| `tests/unit/test_fused_rest_sed.py` | rewrite to use `predict_rest_sed` (now orchestrator-backed) |
| `tests/unit/forward/test_kernel_adapters.py` | **delete** |
| `tests/unit/forward/test_kernel_strategy.py` | **delete** |
| `tests/unit/forward/test_kernel_strategy_classmethods.py` | **delete** if exists |
| `tests/unit/forward/test_kernel_build_log.py` | **delete** |
| `tests/unit/forward/test_strategy_injection.py` | **delete** |
| `tests/unit/test_structural_kernel_cache.py` | rewrite — structural cache survives, just no kernel objects in it |

### Docs

| File | Change |
|---|---|
| `docs/adr/0004-kernel-strategy-module.md` | mark **superseded** by Phase 3, add link to the precomp design |
| `CLAUDE.md` | drop the `mode="_traceable"` reference; document `predict_via_precomp` as the opt-in fast path |
| `AGENTS.md` | update `src/tengri/forward/_kernels/` description (or remove the line) |
| `docs/dev/photometry_path_unification.md` | mark Phase 3d as shipped in the implementing PR |
| `docs/dev/precompute_test_consolidation.md` | mark consolidation as executed |
| `docs/dev/phase3d_temp_plan.md` | **delete this file** in the same PR |

## Acceptance criteria

After the PR:

1. `grep -rn "_kernels" src/ tests/ notebooks/ bench/` returns zero matches in non-historical files.
2. `model.predict_photometry(params)` produces the same output as before (the exact path was already the default for any non-compositional mode, and compositional was bit-exact to it).
3. `model.predict_spectrum(params, wave_obs)` works without `mode=`.
4. `model.predict_observables(params).phot_fnu` matches `model.predict_photometry(params)` bit-exact.
5. `model.observation.predict_via_precomp(state, params).phot_fnu` matches within 0.5% (single-component dust) / 0.3% (two-component dust) — unchanged from Phase 3c-3d-agn.
6. `model.predict(params)` `Prediction` wrapper still produces all derived quantities (`.sfh`, `.sed.dn4000`, `.lines.halpha`, etc.).
7. CI green: lint + smoke + test on 3.11 and 3.12.
8. Notebook 01 runs end-to-end with the new vocabulary.
9. `from tengri import KernelStrategy` raises `ImportError` with the migration message.
10. `SEDModel(approx={"wave_precomp": True})` raises `TypeError` with the legal-values list.
11. `SEDModel(approx=WavePrecomp())` resolves to `wave_precomp=True` internally (and ztable interpolation when redshift is free); `SEDModel()` (no `approx=`) runs the exact wave-grid path with all flags `False`.
12. `SEDModel(approx=WavePrecomp(n_z=200, z_min=0.0, z_max=3.0))` threads `n_z` / `z_min` / `z_max` into the stellar component's `redshift_spec`.
13. `model.spec.summary()` shows the resolved method in human-readable form: `approx: wave_precomp (ztable: enabled for free z)` when active, `approx: none (exact)` by default.

## Risk + mitigation

- **`predict_rest_sed` was JIT'd via compositional kernel.** Now goes through `predict_via_orchestrator` which is already JIT-compatible (it's what `predict_via_precomp` uses). No perf regression expected for warm calls.
- **`predict_spectrum` chunking.** `_observe_spectrum_from_rest_sed_chunked` is the only thing that chunks; reimplement it inline using `compute_spectrum` over chunks. Same math.
- **Test count drops.** ~7 test files deleted. That's fine — they tested deleted code. The behaviour they pinned is covered by the precomp tolerance tests and the new exact-only path.
- **ADR-0004 supersession.** The ADR documented `KernelStrategy` as the public flexibility seam. Removing it is a breaking change. Mitigated by the deprecation `ImportError` with explicit migration text.

## What this PR is NOT

- Not adding spectrum to `predict_via_precomp`. Spectrum stays exact through `observation.predict`. (Phase 3c-3-spec follow-up if anyone wants it.)
- Not removing the channel methods (`predict_photometry`, `predict_spectrum`, …). They stay as the user-facing API; just lose `mode=`.
- Not flipping `observation.predict` default to precomp (Phase 3c-3e rejected; documented).
- Not touching `compile_signature()` for cross-galaxy reuse (Phase 4).
