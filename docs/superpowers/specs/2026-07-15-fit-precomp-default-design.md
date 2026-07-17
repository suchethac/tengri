# Design: precomp-by-default for fits (+ auto-prewarm)

**Date:** 2026-07-15
**Status:** proposed
**Area:** `area:inference`, `area:api`, `area:perf`; `breaking-change`

## Problem

Building a `SEDModel` defaults to `approx=None` (the exact wave-grid forward
path) — correct, but ~14–27× slower per forward evaluation than the precompute
LUT path (`approx=WavePrecomp()` / `SpectrumPrecomp()`). Because every inference
backend evaluates the forward model inside its loss/gradient, that slowdown
flows straight into fits.

Today the LUT is opt-in at *build* time, so users (and every gallery example)
must know to pass `approx=WavePrecomp()` to get a fast fit — and if they do, the
model's *prediction/mock/"truth"* also becomes approximate, not just the fit.
Meanwhile `Fitter._auto_precompute_photometry` fires only the *legacy*
`ensure_photometry_precomputed` path, which does **not** deliver the modern
`approx=` LUT speedup (measured: an exact-built photometry fit is still ~26×
slower per value+grad than the WavePrecomp path).

Separately, `Fitter.run()` compiles lazily on first call, so the JIT compile is
paid inside the (timed) fit rather than as a distinct up-front step; prewarming
is opt-in via `fitter.prewarm(...)` / `fitter.compile(...)`.

## Goals

1. **Model build / prediction stay exact by default.** `approx=None` remains the
   default for `SEDModel.build` / `SEDModel(...)`; prediction, mocks, and plotted
   "truth" are unapproximated unless the user explicitly opts in.
2. **Fits default to precomp**, auto-selected by data type, with a clean opt-out.
3. **JIT compile happens before the fit** (auto-prewarm), populating the
   persistent cache and giving a warm sampling loop, without a manual step.
4. No new approximation vocabulary — reuse the existing `approx=` config objects
   and the content-hashed `tengri_precomp` cache.

## Non-goals

- Changing the exact-path numerics or any `approx=` LUT internals.
- Changing build-time defaults for prediction.
- A new precompute implementation — this only *routes* fits through the existing
  validated one.

## Approach

**Chosen: clone the model with the selected `approx` and fit the clone.**
Reuses the entire validated `approx=` machinery and the precomp cache instead of
duplicating LUT logic in the inference layer.

*Alternative considered — reroute the loss to a LUT without rebuilding the
model:* rejected. It would fork the forward path inside `inference/`, duplicating
the projection/precompute logic that already lives behind `approx=`, and risk
drift from the build-time path the LUT was validated against.

## Components

### 1. `SEDModel.with_approx(approx)` → `SEDModel`

Returns a new model cloned from `self` with a different `approx` (or `None` for
exact). Reuses the stored build inputs: `spec`, `ssp_data`, `observation`,
`_agn_config`, `_forward_dtype`, `_wave_chunk_size`, and `csp_integration`.
Building the clone builds only the LUT, which the `tengri_precomp` cache
persists (content-hashed on SSP grid × filters × z-grid), so repeat fits are
cheap. If `approx` equals the model's current effective config, returns `self`
(no rebuild).

`ForwardModel.with_approx(approx)` mirrors it: clone `sed` via
`sed.with_approx(...)` and re-wrap, preserving the ForwardModel's own state.

**Interface contract:** input a precomp config / tuple / `None`; output a new
model whose forward path uses that config; depends only on attributes already
stored at construction. Independently testable: build exact, `with_approx(...)`,
assert `predict_photometry` matches the directly-built precomp model.

### 2. `approx=` on the fit entry points

`Fitter.__init__(..., approx="auto")`, threaded through `ForwardModel.fit(...)`,
`SEDModel.fit(...)`, and the population/spatial fit shortcuts.

Resolution at Fitter construction, before the loss is built:

- `"auto"` (default): if the model already carries a modern `approx` (built with
  one) → use the model as-is. Otherwise clone via `with_approx(cfg_auto)` where
  `cfg_auto` is the data-type default (table below).
- `None`: force the **exact** wave-grid path — clone to exact if the model was
  built with an `approx`; else use as-is. (Reproduces pre-change behavior for an
  exact-built model.)
- explicit `WavePrecomp(...)` / `SpectrumPrecomp(...)` / `FeaturePrecomp(...)` /
  tuple: clone with exactly that (overrides build-time approx).

### 3. `"auto"` → config mapping

Keyed on the Fitter's resolved `data_type` and whether emission lines are fit
(existing `self._eline_marginalize` / `self._eline_fitted`):

| data_type | no lines | lines fit |
|---|---|---|
| `photometry` | `WavePrecomp()` | `(WavePrecomp(), FeaturePrecomp())` |
| `spectroscopy` / `joint` | `SpectrumPrecomp()` | `(SpectrumPrecomp(), FeaturePrecomp())` |

Config objects use their own defaults (z-grid sampling etc.); users override by
passing an explicit config instead of `"auto"`.

### 4. Legacy `_auto_precompute_photometry` reconciliation

When a modern `approx` is active on the (possibly cloned) fit model, the legacy
`ensure_photometry_precomputed` path is skipped, so there is no double-precompute
and no silent legacy/modern conflict. The legacy hook remains only for the
`approx=None` (explicit exact) fit path, unchanged.

### 5. Auto-prewarm (loss + sampler + predict surface)

`Fitter.run(..., prewarm=True)` (default), also exposed on `forward.fit(...)`.
Before entering the sampling/optimization loop, compile and block:
- the loss + gradient (`_get_or_build_loss_fn` / `_get_or_build_grad_fn`),
- the selected method's sampler kernel (reusing `prewarm()` / `compile()` for
  that method), and
- the post-fit **predict surface** on the fit model: `predict_photometry` and
  `predict_properties` (the JIT/vmap-safe accessors used for posterior-predictive
  checks and derived-quantity roll-ups). Both route through the same
  `predict_observables_jit` orchestrator, so warming them is one blocking call
  each on the init-param structure.

Effect: the JIT compile is a distinct up-front step, the persistent JAX cache is
populated, and both the fit loop **and** the immediate post-fit exploration run
warm. `prewarm=False` restores lazy compile-on-first-call. Redundant prewarm is
cheap (cached), matching the existing `prewarm()` contract. Subsumes the manual
`fitter.prewarm(...)` calls in the spine notebooks.

### 5b. Predict surface — JIT/LUT facts this relies on

Per the `SEDModel` class contract, **all** prediction methods are JIT/vmap-safe
and gradient-safe **except `predict()`** (the rich, lazy, cached exploration
object — not a pure array→array function, so intentionally not jittable; its
lean twins are `predict_photometry` / `predict_properties` / `predict_spectrum`
/ `predict_*_quantities`). They route through `predict_observables_jit`.

Two consequences the prewarm depends on:
- **Use `predict_photometry`, not `predict_observables`, for warm/fast
  posterior-predictive.** `predict_observables` bypasses the WavePrecomp LUT
  (measured 16.5×); `predict_photometry` honors it. The prewarm warms the
  LUT-honoring surface. (The spine notebooks currently chunk-vmap
  `predict_observables` as a memory workaround; that becomes unnecessary on the
  LUT path — a follow-up, not part of this PR.)
- **The returned `Posterior` references the fit model.** Under `approx="auto"`
  the fit model is the LUT clone, so `posterior`-driven `predict_photometry` is
  warm and fast. The user's *original* (exact) model object is unchanged
  (Goal 1). This is documented so the identity difference is not surprising.

## Data flow

```
forward.fit(data, noise, method, approx="auto", prewarm=True)
  └─ Fitter(model, data, noise, approx="auto")
        ├─ resolve data_type (+ line flags)
        ├─ resolve effective approx  ("auto" → cfg_auto | None → exact | cfg)
        ├─ fit_model = model.with_approx(cfg) if a clone is needed else model
        ├─ build loss/grad from fit_model
        └─ .run(method, prewarm=True)
              ├─ if prewarm: compile+block loss/grad + sampler kernel
              │             + predict_photometry + predict_properties (fit model)
              └─ run sampling/optimization loop (warm)
                    → Posterior(_model = fit_model)  # LUT clone under "auto"
```

## Behavioral change

Fits become approximate by default. Validated far below noise in this campaign:
photometry at SNR 20 → posterior-mean shift 0.035–0.064 σ vs exact; z=4
Lyman-break/IGM → max band deviation 0.006% (noise floor 6.67%), identical MAP.
Handling (per decision): **silent at runtime**, documented in `fit()` docstrings
+ a `changelog` entry, labeled `breaking-change`. Escape hatch: `approx=None`.

## Testing

- **Agreement (contract):** for photometry / spectroscopy / joint / joint+lines,
  `fit(approx="auto")` vs `fit(approx=None)` posteriors agree within a small
  σ-fraction on a fixed-seed mock (MAP for speed; one NUTS case at low D).
- **Exact reproduction:** `approx=None` reproduces the pre-change exact result
  bit-for-bit (regression guard).
- **Override honored:** explicit `WavePrecomp(n_z=...)` changes the LUT config.
- **Build-time approx respected:** a model built with `approx=WavePrecomp()` fit
  with `"auto"` is not re-cloned (identity/no extra LUT build).
- **Entry points:** both `SEDModel.fit`/`Fitter(sed_model,...)` and
  `ForwardModel.fit`/`Fitter(forward,...)` reach the same fast path.
- **with_approx unit:** cloned model's `predict_photometry` == directly-built
  precomp model's, and exact when `approx=None`.
- **Prewarm:** with `prewarm=True`, the first post-prewarm loss/grad call issues
  no new compile (compile counter / timing); `prewarm=False` still fits.
- **Predict prewarm:** after a `prewarm=True` fit, the first
  `predict_photometry` and `predict_properties` calls on the returned posterior's
  model issue no new compile; and they use the LUT (fast, not the
  `predict_observables` bypass).
- **Posterior model identity:** under `approx="auto"` the returned
  `posterior._model` is the LUT clone (fast predict); the user's original model
  object is unchanged and still exact.
- **No double-precompute:** legacy `ensure_photometry_precomputed` is not invoked
  when a modern approx is active.

## Rollout / PR structure

Single PR on `worktree-docs-precompute-sweep`:
1. `SEDModel.with_approx` / `ForwardModel.with_approx`.
2. `Fitter` `approx="auto"` resolution + `"auto"` mapping + legacy reconciliation.
3. `prewarm=True` default on `run()` / `fit()`.
4. Tests (above).
5. Docstrings + changelog (`breaking-change`).
6. **Revert PR #1180's example edits** — under Goal 1 the gallery builds exact and
   gets its fit speedup from this feature; drop the manual spine `fitter.prewarm`
   calls.

## Risks

- **Faithful clone:** `with_approx` must pass through every build input that
  affects the forward path. Mitigation: the `with_approx` unit test asserts
  equality against a directly-built precomp model; if any input is missed, the
  prediction diverges and the test fails.
- **Approximation-by-default surprising a user:** mitigated by the documented
  `approx=None` escape hatch and the validation evidence; no silent numeric
  change to the exact path itself.
- **Prewarm cost for heavy methods (NUTS AOT ~20 s):** it's the same compile the
  fit would pay anyway, just up-front and cached; `prewarm=False` opts out.
