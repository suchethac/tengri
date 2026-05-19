# Unify forward-projection paths under `state → observation.predict()` + three modes

**Status:** Draft for review
**Date:** 2026-05-19
**Supersedes:** `docs/dev/predict_consolidation_design.md` (delete before review).
**Builds on:**
- [`three_evaluation_modes.md`](three_evaluation_modes.md) — the three-mode contract (one physics, three execution modes), today realised at the **component layer**.
- [`optimization-architecture.md`](optimization-architecture.md) — today's per-observable benchmark numbers.
- ADR-0004 (kernel-strategy-module) — the strategy/adapter pattern.

## TL;DR

Today tengri has **five public predict methods**
(`predict_photometry`, `predict_spectrum`, `predict_line_fluxes`,
`predict_magnitudes`, `predict_luminosity`) and **seven kernel adapters**
(three pairs for photometry × spectrum across three tiers, plus
exact-rest-SED). Each observable has been wired separately. Joint
photometry+spectrum is **two independent forward passes whose
log-likelihoods are summed at the end** — no fused joint forward pass
exists today.

The `ObservationModel` Protocol scaffold (`protocols/observation.py`)
already defines the right abstraction: `state → observation.predict(state, params) → dict of observables`. **Nothing consumes it.** The active
`observation.Observation` class is a unified container that holds
optional Photometry / Spectroscopy / LineFluxData / SpectralIndexData
sub-blocks and exposes `observe_photometry()` / `observe_spectrum()` —
but the fast kernels bypass it.

This doc proposes closing the gap once: **one physics path
(orchestrator chain), one projection method on `Observation` that
returns whatever observables the observation contains, and three
execution modes that wrap that pair**. The result handles photometry,
spectrum, lines, magnitudes, luminosity, **and joint** uniformly.

## Today's surface — full anatomy

### Public methods on `SEDModel`

| Method | Inputs | Output | Notes |
|---|---|---|---|
| `predict(params)` | params | `Prediction` lazy object | Calls predict_* under the hood |
| `predict_photometry(params, mode="auto", ...)` | params, mode | `(n_filt,)` | 5-mode dispatcher |
| `predict_spectrum(params, wave_obs=..., mode="auto", ...)` | params, mode, wave_obs | `(n_pix,)` | 5-mode dispatcher (mirror of photometry) |
| `predict_line_fluxes(params, ...)` | params, target λs | dict {line → flux} | Single path; runs orchestrator + extracts derived line data |
| `predict_magnitudes(params)` | params | `(n_filt,)` AB | Delegates to `predict_photometry` |
| `predict_luminosity(params)` | params | `(n_wave,)` L_ν | Delegates to `predict_rest_sed` |

Plus the `_via_orchestrator` siblings (`predict_photometry_via_orchestrator`, `predict_spectrum_via_orchestrator`, `predict_line_fluxes_via_orchestrator`, etc.) — used by `inference/loss_functions.py` and `inference/jit_engine.py`.

### Kernel adapters (`forward/_kernels/_adapters.py`)

| Adapter | Product | Compat gate |
|---|---|---|
| `ExactRestSEDKernel` | `rest_sed` | always |
| `CompositionalRestSEDKernel` | `rest_sed` | when rest_wavelength set |
| `CompositionalPhotometryKernel` | `photometry` | rest_wavelength + filters set; blocks on ramp/chem_evol metallicity |
| `CompositionalSpectrumKernel` | `spectrum` | rest_wavelength + (precomputed spectroscopy OR wave_obs) |
| `HybridPhotometryKernel` | `photometry` | precomputed.photometry + z_fixed; blocks on tabulated SFH |
| `HybridPhotometryZTableKernel` | `photometry` | precomputed.photometry_ztable (free z) |
| `HybridSpectrumKernel` | `spectrum` | precomputed.spectroscopy + z_fixed |

**Photometry and spectrum kernels are independent.** Lines / magnitudes / luminosity bypass kernels entirely.

### Distinct physics implementations per observable

| Observable | Physics paths today |
|---|---|
| Photometry | 3 (exact via orchestrator → filter loop, compositional fused kernel, hybrid fused kernel with SSP×filter precompute) |
| Spectrum | 3 (mirror of photometry — exact, compositional, hybrid) |
| Lines | 1 (orchestrator path; nebular component publishes `line_waves`, `line_lums`) |
| Magnitudes | 1 (delegates to photometry) |
| Luminosity | 1 (delegates to rest SED) |

When a component changes (new physics block, new derived key, new param), the compositional and hybrid kernels for photometry **and** for spectrum all need to be updated independently. Drift between them is not caught by `validate_pipeline` — that only checks the orchestrator path. Drift is detected only by snapshot regression tests.

### Joint photometry+spectrum today

`Observation.is_joint` returns `True` when both photometry and spectroscopy sub-blocks are configured. But the **forward pass is not joint**:

- `inference/loss_functions.py:_build_prediction()` (~line 81–88) handles `data_type == "joint"` by calling **both** `model.predict_photometry()` **and** `model.predict_spectrum()` and concatenating the arrays.
- The likelihood combines them by summing log-probabilities per channel.

There is no fused joint kernel. Joint inference today pays two forward-pass costs and two `@jit` traces.

### `ObservationModel` Protocol — the scaffold

`src/tengri/protocols/observation.py` defines:

```python
class ObservationModel(Protocol):
    name: str
    def declared_parameters(self) -> list[ParamSpec]: ...
    def predict(self, state: ForwardState, params: Mapping) -> Mapping[str, Array]:
        """Returns {"phot_fnu": ..., "spec_fnu": ..., "lines_flux": ..., "indices": ...}"""
```

The protocol docstring is explicit: *"Nothing in tengri consumes this protocol yet."* A concrete `PhotometryObservationModel` exists at `observation/photometry_model.py:50` as scaffold. No `JointObservationModel`. No consumption from SEDModel or from inference.

## The general abstraction

Once observation is the projection seam, **the same shape works for every observable, including joint**:

```python
# ONE physics, always:
state = run_components(chain, params)                  # orchestrator

# ONE projection, dispatched by observation contents:
observables = observation.predict(state, params)
# → {"phot_fnu": (n_filt,), "spec_fnu": (n_pix,), "lines": {...}, "magnitudes": ...}
# Joint = observation contains both photometry and spectroscopy sub-blocks,
# so the dict has both keys. No special case.
```

Three execution modes wrap that pair:

```
eager     — call the pair directly, no @jit
jit       — @jit(the pair)
precomp   — @jit(the pair) with one or more components in `chain` swapped
            for precomputed-lookup variants (same Protocol shape, faster apply)
```

`KernelStrategy` (ADR-0004) selects the mode. Every observable inherits all three modes for free because the observation knows how to project the same orchestrator state into whatever data products it carries.

### What this collapses

| Today | After |
|---|---|
| 5 public `predict_*` methods | 1 `model.predict_observables(params, mode="auto")` returning a dict, plus `Prediction` umbrella unchanged |
| 7 kernel adapters | 3 (one per mode, observable-agnostic — they wrap the same physics + projection pair) |
| 3 photometry physics × 3 spectrum physics = 6 hand-written kernels | 1 physics + 1 projection per observation type |
| Joint = two forward passes + concat | Joint = one forward pass; observation projects to both channels |
| Lines / magnitudes / luminosity bypass kernels | Lines / magnitudes / luminosity inherit the same modes because they're entries in `observation.predict()` output |
| `_via_orchestrator` suffix on 8 methods | Suffix dies — there is only the orchestrator path |

## What needs to land

Three phases, each independently shippable. Sequenced for risk minimisation — the riskiest physics work is last.

### Phase 1 — Make `Observation.predict()` the projection seam

**Goal:** Active code starts consuming the `ObservationModel` Protocol. No behaviour change.

- Add `Observation.predict(state, params) → dict` on the existing unified `Observation` class. Internally calls `observe_photometry()` / `observe_spectrum()` / line extraction / magnitude conversion as needed based on which sub-blocks are configured.
- Wire `_predict_photometry_exact` and `predict_photometry_via_orchestrator` (Group A and Group D from the old framing) to both route through `Observation.predict()` and extract the `phot_fnu` key. Same for spectrum.
- The `_via_orchestrator` siblings become thin shims around the unified path.
- **Behaviour:** unchanged. Same arrays out.
- **Effort:** 1 PR, ~half a day.

### Phase 2 — Collapse `compositional` to `@jit(physics + projection)`

**Goal:** The "compositional" kernel becomes `@jit` applied to the same orchestrator + `Observation.predict()` pair. One physics for compositional photometry, compositional spectrum, **and joint**.

- Build `make_jit_observables(model)` → `@jit(lambda params: observation.predict(run_components(...), params))`.
- Snapshot test: assert bit-equivalence with today's `_compositional.photometry` and `_compositional.spectrum` outputs. **If they differ**, decide which is correct (see Q3 below); my default is **orchestrator wins** because `validate_pipeline` enforces its contract.
- Delete `CompositionalPhotometryKernel`, `CompositionalSpectrumKernel`, `CompositionalRestSEDKernel` (or reduce them to one strategy entry).
- Joint inference automatically benefits: the single `@jit` trace yields both `phot_fnu` and `spec_fnu` from one forward pass.
- **Behaviour change risk:** medium — snapshot tests must regress. Inference numerics must remain bit-equivalent within tolerance.
- **Effort:** 1 PR, ~1 day for photometry+spectrum together (no reason to separate them — the unified projection means they collapse simultaneously).

### Phase 3 — Re-express `hybrid` as "swap one component for a precomputed variant"

**Goal:** The "hybrid" mode becomes the same `@jit(physics + projection)` chain but with the stellar component swapped for `StellarPrecomputedSEDComponent` (a precompute-lookup variant). Same chain semantics, same publishes/requires, different `apply`.

- Build `StellarPrecomputedSEDComponent` implementing `SEDComponent` with `publishes(lnu_age, sfr, log_mstar, …)` matching `StellarSEDComponent`. `apply()` performs the precompute lookup instead of the full CSP einsum.
- Build a model-construction switch (`SEDModel.build(..., mode="precomputed")` or a `KernelStrategy` policy) that selects the precomputed variant when the model is built.
- All seven downstream observables (photometry, spectrum, lines, magnitudes, luminosity, joint) inherit the precompute speedup for free, because they all flow through `state.derived["lnu_age"]` etc.
- Delete `HybridPhotometryKernel`, `HybridSpectrumKernel`, `HybridPhotometryZTableKernel`.
- **Behaviour change risk:** highest — historical hybrid path had 0.02–0.33% error vs exact. Equivalent error tolerance must hold. Tests + benchmarks regress against today's hybrid numbers.
- **Effort:** 1–2 PRs, ~1 week.

### Names that fall out

- `predict_photometry_via_orchestrator`, `predict_spectrum_via_orchestrator`, and the other `_via_orchestrator` siblings disappear (Phase 1).
- The 5 `predict_*` public methods collapse to `model.predict_observables(params, mode="auto")` returning the dict + `model.predict(params)` returning `Prediction` (unchanged).
- The 7 kernel adapter classes collapse to 3 (one per mode), and they become observable-agnostic.
- `KernelStrategy.select(...)` returns a mode (`eager` / `jit` / `precomp`), not a `(mode, observable)` pair.

## What this means for the in-flight PRs

- **PR #106 (rename `_=` → `defaults=`).** Independent of this work. Proceed with `defaults=` once Q1 confirmed.
- **PR #109 (predict_* docstring sign-posting).** Land as-is — sign-posting is useful in the interim.
- **PR β as originally scoped** (deprecate `_via_orchestrator` only). **Stop.** Re-scope as Phase 1 of this plan.
- **`predict_consolidation_design.md`** — supersedes by this file. Delete before review.

## Open questions

1. **`defaults` or `default` for PR #106?** I'd been going with `defaults` (plural) — what you said. `default` (singular) matches `dict.get`. **Confirm before PR #106 proceeds.**
2. **Phase 1 surviving name** for the merged `_predict_photometry_exact` + `predict_photometry_via_orchestrator`? My default: keep `predict_photometry(mode="exact")` as public; route through `Observation.predict()` internally.
3. **Phase 2 tie-breaker — if compositional ≠ orchestrator (bit-level)?** My default: **orchestrator wins** (it's the path `validate_pipeline` enforces). Any compositional-specific optimisations get re-introduced as graph transformations on the unified chain.
4. **Joint forward-pass guarantee.** Phase 2 fuses joint photometry+spectrum into one forward pass. Inference benchmarks need to confirm this doesn't regress vs today's two-pass approach (today: 2 traces, 2 JITs; after Phase 2: 1 trace, 1 JIT, but a larger graph). Snapshot performance test before/after.
5. **Effort vs Paper II priorities.** This is ~2 weeks of refactor work with no new physics. Is now the right time, or does it wait until after Paper I notebook series stabilises?

## Decision needed before any code

- [ ] Approve the three-phase plan as scoped?
- [ ] Pick Q1 (`defaults` vs `default`)?
- [ ] Pre-decide Q3 (orchestrator wins if compositional diverges)?
- [ ] Confirm joint forward-pass goal for Phase 2 (single fused trace)?
- [ ] Confirm sequencing: photometry+spectrum collapse together (not staggered)?
