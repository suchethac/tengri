# Unify forward-projection paths under `state → observation.predict() → Observables`

**Status:** Draft for review — revision 2 (2026-05-19)
**Supersedes:** `docs/dev/predict_consolidation_design.md` (delete before review) and revision 1 of this file.
**Builds on:**
- [`three_evaluation_modes.md`](three_evaluation_modes.md) — the three-mode contract (one physics, three execution modes), today realised at the **component layer**.
- [`optimization-architecture.md`](optimization-architecture.md) — today's per-observable benchmark numbers.
- ADR-0004 (kernel-strategy-module) — the strategy/adapter pattern.
- ADR-0009 (typed-pipeline-contract) — `publishes` / `requires` and `validate_pipeline`.

## TL;DR

Today tengri has **five public predict methods** (`predict_photometry`, `predict_spectrum`, `predict_line_fluxes`, `predict_magnitudes`, `predict_luminosity`), **seven kernel adapters**, and a `mode=` kwarg that smashes physics-approximation choice + JIT-wrapping choice + path selection into one overloaded string. There is no fused joint forward pass; joint inference pays two `@jit` traces.

**The redesign:**

1. **`Observation.predict(state, params) → Observables`** is the single projection seam. `Observables` is a **NamedTuple synthesised per model** at construction time, with attribute access (no dicts).
2. **Two orthogonal build-time knobs** replace the `mode=` kwarg:
   - `compile=` controls **JIT wrapping** (`per_component` / `fused`).
   - `approx=` controls **which approximate component variants enter the chain** (a dict — `{"wave_precomp": True, "ztable": True, …}` — extensible to future approximations).
3. **Per-component JIT by default.** Cold-start drops from one 30–75 s fused compile to many ~0.5 s component compiles; the persistent disk cache picks each up independently. `compile="fused"` is opt-in for hot inference loops.
4. **Approximations live with the physics.** Each `SEDComponent.precompute()` returns the regular `SEDComponentState` plus, when its approximation flag is set, the lookup tables its own `apply()` uses on the fast path. **No `*Precomputed` sibling classes, no duplicate code.**
5. **Channel methods** (`predict_photometry`, `predict_spectrum`, …) survive as one-line aliases that pull a field off the `Observables` NamedTuple. `Observation` ↔ `Observables` is a direct mirror.

## Today's surface — full anatomy

### Public methods on `SEDModel`

| Method | Inputs | Output | Notes |
|---|---|---|---|
| `predict(params)` | params | `Prediction` lazy object | Interactive introspection — `sfh`, `sed`, `lines`, … |
| `predict_photometry(params, mode="auto", ...)` | params, mode | `(n_filt,)` | 5-mode dispatcher |
| `predict_spectrum(params, wave_obs=..., mode="auto", ...)` | params, mode, wave_obs | `(n_pix,)` | 5-mode dispatcher (mirror of photometry) |
| `predict_line_fluxes(params, ...)` | params, target λs | dict {line → flux} | Single path; runs orchestrator + extracts derived line data |
| `predict_magnitudes(params)` | params | `(n_filt,)` AB | Delegates to `predict_photometry` |
| `predict_luminosity(params)` | params | `(n_wave,)` L_ν | Delegates to `predict_rest_sed` |

Plus the `_via_orchestrator` siblings (now thin shims after Phase 1).

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

### Distinct physics implementations per observable

| Observable | Physics paths today |
|---|---|
| Photometry | 3 (exact via orchestrator → filter loop, compositional fused kernel, hybrid fused kernel with SSP×filter precompute) |
| Spectrum | 3 (mirror of photometry — exact, compositional, hybrid) |
| Lines | 1 (orchestrator path; nebular component publishes `line_waves`, `line_lums`) |
| Magnitudes | 1 (delegates to photometry) |
| Luminosity | 1 (delegates to rest SED) |

When a component changes (new physics block, new derived key, new param), the compositional and hybrid kernels for photometry **and** for spectrum all need to be updated independently. Drift between them is not caught by `validate_pipeline` — that only checks the orchestrator path.

### Joint photometry+spectrum today

`Observation.is_joint` returns `True` when both photometry and spectroscopy sub-blocks are configured. But the **forward pass is not joint**:

- `inference/loss_functions.py:_build_prediction()` (~line 81–88) handles `data_type == "joint"` by calling **both** `model.predict_photometry()` **and** `model.predict_spectrum()` and concatenating the arrays.
- The likelihood combines them by summing log-probabilities per channel.

There is no fused joint kernel. Joint inference today pays two forward-pass costs and two `@jit` traces.

## The new architecture

### The container — `Observables` NamedTuple, synthesised per model

`Observables` is the dual of `Observation`. Built once at `SEDModel.__init__`:

```python
# Inside SEDModel.__init__:
fields: list[tuple[str, type]] = []
if observation.can_do_photometry:
    fields += [
        ("phot_fnu",      jnp.ndarray),   # observed F_nu [erg/s/cm²/Hz]
        ("phot_rest_fnu", jnp.ndarray),   # rest-frame F_nu at d_L=10pc, same filters
    ]
if observation.can_do_spectroscopy:
    fields += [("spec_fnu", jnp.ndarray)]
if observation.has_line_fluxes:
    fields += [("lines_flux", jnp.ndarray)]
if observation.has_spectral_indices:
    fields += [("indices", jnp.ndarray)]
self._Observables = NamedTuple("Observables", fields)
```

User-facing:
```python
o = model.predict_observables(params)
o.phot_fnu               # array, shape (n_filt,)
o.spec_fnu               # array, shape (n_pix,)
o.mag_apparent           # @property → AB mag from o.phot_fnu
o.mag_absolute           # @property → AB mag from o.phot_rest_fnu
o.lines_flux             # array, shape (n_lines,)
```

Direct mirror:

| Observation contains | Observables field(s) exist |
|---|---|
| `Photometry` | `.phot_fnu` `.phot_rest_fnu` (+ `.mag_apparent` / `.mag_absolute` properties) |
| `Spectroscopy` | `.spec_fnu` |
| `LineFluxData` | `.lines_flux` |
| `SpectralIndexData` | `.indices` |

If a sub-block is absent the corresponding field **does not exist** on the NamedTuple — `AttributeError` on access, not silent zero. Pytree registration is automatic.

Absolute magnitudes are computed by re-projecting `state.sed_intrinsic` through the same filters at `z=0, d_L=10pc` — physically correct independent of K-correction debates. Cost: doubles the filter integrals (small fraction of the forward pass).

### The general projection

Once observation is the projection seam, the same shape works for every observable, including joint:

```python
# ONE physics, always:
state = run_components(chain, params)                  # orchestrator

# ONE projection, dispatched by observation contents:
observables = observation.predict(state, params)
# → Observables(phot_fnu=..., phot_rest_fnu=..., spec_fnu=..., lines_flux=...)
# Joint = observation has both sub-blocks, so both fields populated. No special case.
```

### Two orthogonal build-time axes — `compile=` × `approx=`

```python
model = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    compile="per_component",         # | "fused" | "auto"
    approx={                         # component-scoped flags; extensible
        "ztable":      True,           # stellar/phot — free-z SSP×filter LUT (shipped)
        "wave_precomp": False,         # stellar/phot — fixed-z SSP×filter LUT (Phase 3)
        # other components add their own flag names as approximations land
    },
    **recipes.star_forming_photometry(),
)
o = model.predict_observables(params)         # always returns Observables NamedTuple
```

#### `compile=` — controls only JIT wrapping

| Value | Effect | Cold | Warm | When |
|---|---|---|---|---|
| `per_component` (default) | Each `SEDComponent.apply` is `@jit`'d independently. `predict_observables` itself is NOT outer-JIT'd; the chain dispatcher runs at Python level. | ~5 s (many ~0.5 s pieces) | ~0.5–0.7 ms | Notebook iteration, schema changes |
| `fused` | One outer `@jit` over `observation.predict ∘ run_components`. Same path as this PR's `predict_observables_jit`. | ~10–30 s | ~0.15–0.3 ms | Inference loops, population fits |
| `auto` | Heuristic: per_component on first call, switches to fused after N evaluations. Stub initially. | varies | varies | Default for users who don't care |

**Per-component JIT only helps when there's no outer `@jit` re-inlining everything** — that's the whole point. Compile cost goes from one 30 s graph to many independent ones, the persistent disk cache picks up each on its own structural fingerprint, and unchanged components survive model edits.

#### `approx=` — controls which approximate component variants enter the chain

Each approximation is a **flag in the dict**, owned by the components it touches. No `*Precomputed` sibling classes. Approximations live with the physics they approximate.

```python
# In components/stellar/component.py — single class, two paths:
@dataclass(frozen=True)
class StellarSEDComponent:
    config: StellarSEDComponentConfig

    def precompute(self, ssp_data, wave_grid, *, approx=None):
        approx = approx or {}
        state = StellarSEDComponentState(
            ssp_grid=ssp_data.ssp_flux,          # always
            wave_rest=wave_grid,
        )
        if approx.get("wave_precomp"):
            state = dataclasses.replace(
                state,
                ssp_filter_lut=_build_ssp_filter_lut(ssp_data, self.config.filters),
            )
        return state

    def apply(self, pipeline_state, params):
        if self._state.ssp_filter_lut is not None:
            return self._apply_via_lut(pipeline_state, params)
        return self._apply_exact(pipeline_state, params)
```

**Each component owns its own approx-flag names.** No global umbrella — flags are component-scoped so their meaning stays sharp and adding a new approximation never collides with an old one:

**Each component owns its own approx-flag names.** No global umbrella — flags are component-scoped so their meaning stays sharp and adding a new approximation never collides with an old one:

| Flag | Owner | Depends on | What it precomputes |
|---|---|---|---|
| `wave_precomp` | stellar (photometry) | — | SSP × filter inner products on a wave grid (the historical "hybrid photometry" math). Valid only when redshift is `Fixed` unless paired with `ztable`. |
| `ztable` | stellar (photometry) | **requires `wave_precomp=True`** | Indexes the SSP × filter integrals on a redshift grid — makes the precompute valid for free redshift. **Auto-enabled when `wave_precomp=True` AND redshift is free.** |
| _(future)_ `dust_precomp` | dust | — | Attenuation curve LUT for a fixed wave grid |
| _(future)_ `cue_interp` | nebular | — | Cue training-grid interpolation tables |
| _(future)_ `dust_taylor` | dust | — | Perturbation expansion in τ |

**Dependency / auto-resolution rules** (run at build time, surfaced via explicit error or log line):

- `ztable=True` with `wave_precomp=False` → **raise** with diagnostic. `ztable` is the free-z variant of `wave_precomp`; it has nothing to index without it.
- `wave_precomp=True` with **free redshift** and `ztable=False` (or unspecified) → **auto-enable `ztable=True`** and log a single line. Without `ztable` the LUT is keyed on a fixed z and would silently give wrong fluxes; safer to upgrade than to raise.
- `wave_precomp=True` with **fixed redshift** → `ztable` is irrelevant; default-off, raise if user explicitly sets `True` with a hint to drop it.
- Unknown flag names → **raise** at build time with the list of legal flags.

Flags otherwise compose — adding `dust_precomp` later is independent of `wave_precomp`/`ztable`. Each component decides what to do with the flags it owns.

#### The matrix

|  | `approx={}` | `approx={"wave_precomp": True}` | `approx={"wave_precomp": True, "ztable": True}` |
|---|---|---|---|
| `compile="per_component"` | bit-exact, ~5 s cold, ~0.7 ms warm | approx-exact, ~5 s cold, ~0.5 ms warm | approx-exact, ~5 s cold, ~0.4 ms warm |
| `compile="fused"` | bit-exact, ~30 s cold, ~0.3 ms warm | approx-exact, ~30 s cold, ~0.18 ms warm | **approx-exact, ~30 s cold, ~0.15 ms warm — fastest** |

The two axes compose freely. Adding a new approx method later is **one entry in the dict + one branch in the relevant component's `precompute` / `apply`** — no surface change, no new method, no new kernel class.

### What this collapses

| Today | After |
|---|---|
| 5 public `predict_*` methods + dict-typed plan | 1 `model.predict_observables(params) -> Observables` NamedTuple + `Prediction` umbrella unchanged + channel methods as one-line aliases |
| 7 kernel adapters | 0 — adapters die. Each component owns its own approximation under the same `SEDComponent` Protocol. |
| 3 photometry physics × 3 spectrum physics = 6 hand-written kernels | 1 physics + 1 projection per observation type |
| Joint = two forward passes + concat | Joint = one forward pass; observation projects to both channels |
| Lines / magnitudes / luminosity bypass kernels | All observables flow through `observation.predict()` |
| `_via_orchestrator` suffix on 8 methods | Suffix dies — there is only the orchestrator path |
| `mode=` kwarg with 5 overloaded values | `mode=` dies; replaced by build-time `compile=` × `approx=` |

## What needs to land

Four phases, each independently shippable. Phase 1 is **done** (PR #112).

### Phase 1 — `Observation.predict()` is the projection seam ✅ shipped

**Status:** PR #112 (cs/cmpoutation-path-clean) open for review.

- `Observation.predict(state, params, *, wave_obs, sigma_v_kms, lsf_*) -> dict` added.
- `predict_photometry_via_orchestrator` and `predict_spectrum_via_orchestrator` reduced to two-line shims that route through the seam.
- Latent bug fixed: `_via_orchestrator` paths now merge `spec.get_fixed_values()` into the projection params dict (previously `params.get("redshift", 0.0)` returned 0.0 when redshift was `Fixed`, producing `dl_cm = inf`).
- Additive: `predict_observables(params)` and `predict_observables_jit(params)` introduced as a preview of the new surface. Currently dict-returning; **Phase 2 swaps to `Observables` NamedTuple.**

### Phase 2 — `Observables` NamedTuple + `compile=` knob ✅ shipped

**Status:** PR #114 (cs/phase2-observables-tuple) open for review, stacked on #112.

What landed:

- `Observables` NamedTuple synthesised in `SEDModel.__init__` from `observation` contents (`src/tengri/observation/observables.py:build_observables_class`). Fields exist iff the observation carries the corresponding sub-block — `AttributeError` on missing channels, not silent zeros. `mag_apparent` / `mag_absolute` `@property`s compute AB mags from `phot_fnu` and `phot_rest_fnu`. Pytree-registered.
- `Observation.predict(state, params, *, observables_type=None)` — returns `Observables` instance when type provided; falls back to the Phase 1 dict otherwise (backward compat for callers that haven't migrated).
- `phot_rest_fnu` computed by re-projection at `z=0, d_L=10 pc` through the same filters (no K-correction debate).
- `compile=` build-time kwarg with values `"per_component"` (default) / `"fused"` / `"auto"` (stub, resolves to `per_component`).
- `approx=` validation + auto-resolution per the dependency table.
- `compile_signature()` includes resolved `compile=` and `approx=` so cache slots stay distinct.
- `SEDModel.Observables` property exposes the synthesised type for advanced users.

What is **NOT** in Phase 2:

- Channel methods (`predict_photometry`, `predict_spectrum`) still have their existing implementations. Routing them through `Observables` and adding the `mode=` `DeprecationWarning` is **Phase 3** alongside the per-component approx implementations.
- `approx={"wave_precomp": True}` is accepted by validation but raises `NotImplementedError` at the per-component apply path — the stellar component's wave-precompute branch lands in Phase 3.
- `compile="auto"` is a stub that resolves to `"per_component"` with a TODO comment.

Bit-exactness verified:

- `predict_observables(p).phot_fnu` vs `predict_photometry_via_orchestrator(p)`: **max diff = 0.0**.
- `predict_observables(p)` vs `predict_observables_jit(p)`: **max diff = 1.15e-41**.
- 394 broad-slice unit tests pass with zero new failures vs Phase 1.

### Phase 3 — `approx={"wave_precomp": True}` — per-component precompute

**Goal:** Each `SEDComponent` that has a precomputable wave-grid lookup grows a single optional branch in `precompute()` / `apply()`, gated by a flag in the `approx` dict. **No new component classes.**

- Extend `SEDComponentState` (in `protocols/component.py`) with optional `precomputed_lut: dict | None` field.
- `StellarSEDComponent.precompute(..., approx={"wave_precomp": True})` builds the SSP×filter LUT and stores it on the returned state; `apply()` reads `self._state.precomputed_lut` and switches paths.
- Same surgical change in **dust** (attenuation curve LUT), **nebular** (Cue grid interp tables when applicable), and **igm** (already has `ztable`).
- `validate_pipeline` learns about the new state field and ensures all consumers see the same publishes/requires regardless of approx setting.
- Delete `HybridPhotometryKernel`, `HybridSpectrumKernel`, `HybridPhotometryZTableKernel`, and the `CompositionalPhotometryKernel` / `CompositionalSpectrumKernel`. They were placeholders for what becomes per-component machinery.
- Move the historical hybrid math **into the components** — no separate kernel files. The `forward/_kernels/_adapters.py` directory shrinks to zero content (the file can be deleted).
- Compile-signature audit: `approx` resolved values get hashed into the signature so two models with different approx settings get different cache slots.
- **Behaviour change risk:** highest — historical hybrid path had 0.02–0.33 % error vs exact. Equivalent error tolerance must hold per component.
- **Effort:** 2–3 PRs (one per component family), ~1–2 weeks.

### Phase 4 — `compile_signature()` drops per-galaxy state

**Goal:** Cross-galaxy compile reuse works regardless of per-galaxy `Fixed` values. Audit deliverable from PR #112's companion finding.

- Drop `z_fixed` and `spec_fixed_id` from `compile_signature()` once the unified path stops closing over fixed values at build time (Phase 2/3 dependency).
- Add `tests/unit/test_compile_signature_cross_galaxy.py` asserting two models with identical physics but different per-galaxy `Fixed(redshift)` share a signature.
- Validation: `approx={"wave_precomp"}` with free z requires `approx={"ztable": True}` too; raise with a diagnostic at build time if a user disables both.
- **Effort:** 1 PR, ~half a day after Phase 3 ships.

### Surface after all phases land

Three concentric methods survive on `SEDModel`. Everything else collapses upward.

| Method | Returns | Purpose | Status |
|---|---|---|---|
| `model.predict_via_orchestrator(params)` | `ForwardState` (raw chain output) | Advanced/debug — custom projections, derived-key inspection, writing your own loss function | **survives** all phases. Rename to `model.forward(params)` is a separate bikeshed; the surface stays. |
| `model.predict_observables(params)` | `Observables` NamedTuple | Hot path — likelihood-facing, JIT'd via `compile="fused"` | new in Phase 2 |
| `model.predict(params)` | `Prediction` lazy wrapper | Interactive — `.sfh.stellar_mass`, `.lines.halpha`, etc. | unchanged across all phases |

Everything that dies:

- `predict_photometry_via_orchestrator`, `predict_spectrum_via_orchestrator`, `predict_line_fluxes_via_orchestrator`, `predict_emission_lines_via_orchestrator` — thin shims after Phase 1; **deleted in Phase 3.** The `_via_orchestrator` suffix has no meaning once there's only one path.
- The 5 `predict_*` public methods (`predict_photometry`, `predict_spectrum`, `predict_line_fluxes`, `predict_magnitudes`, `predict_luminosity`) become one-line aliases off `Observables` in Phase 3: `predict_photometry(params) -> predict_observables(params).phot_fnu`. The aliases can be deprecated later if the migration is smooth.
- `mode=` kwarg: deprecation warning in Phase 3, deleted in a subsequent release.
- 7 kernel adapter classes: **all deleted in Phase 3.** Their math moves into the components they approximate. `forward/_kernels/_adapters.py` ends Phase 3 empty; the file is deleted.
- `predict_observables_jit`: name dies. The choice of JIT wrapping is `compile=` at build time, not a method-name suffix. (Phase 2 still exposes it as a transitional convenience; deletion is Phase 3.)

**Three methods, three purposes.** That's the steady state.

## What this means for the in-flight PRs

- **PR #112 (Phase 1).** Merged. Phase 2 amends `predict_observables` to return `Observables` NamedTuple.
- **PR #106 (rename `_=` → `defaults=`).** Independent of this work.
- **PR #109 (predict_* docstring sign-posting).** Land as-is — sign-posting is useful in the interim.
- **`predict_consolidation_design.md`** — superseded by this file. Delete before review.

## Open questions

1. **Phase 2 tie-breaker — if `compile="fused"` ≠ `compile="per_component"` (bit-level)?** Default: **`per_component` wins** (it's the canonical orchestrator path that `validate_pipeline` enforces). Any per-component-specific optimisations get re-introduced as graph transformations on the unified chain.
2. **Joint forward-pass guarantee.** Phase 2 fuses joint photometry+spectrum into one trace. Inference benchmarks need to confirm this doesn't regress vs today's two-pass approach. Snapshot performance test before/after.
3. **`approx` validation policy.** Resolved 2026-05-19. `ztable` requires `wave_precomp`; raise if violated. `wave_precomp` with free redshift auto-enables `ztable` with a log line; raise on unknown flag names. See the dependency table above.
4. **`mag_absolute` rest-frame filter choice.** Re-projection at z=0 uses the same filter curves as the apparent mag. For high-z sources this means the "absolute mag" is in the rest-frame equivalent of the observed-frame filter — astrophysically meaningful only when the user understands which rest-frame band that corresponds to. Should we ship a parallel set of `rest_filters=` at build time for explicit rest-frame band choice?
5. **Effort vs Paper II priorities.** Phase 2 + 3 + 4 is ~3 weeks of refactor work with no new physics. Is now the right time, or does it wait until after Paper I notebook series stabilises?

## Decision needed before any code

- [x] Approve the four-phase plan as scoped? — Phase 1 already shipped.
- [x] `compile=` and `approx=` as the two build-time knobs. Confirmed 2026-05-19.
- [x] `wave_precomp` = photometry SSP × filter LUT, owned by stellar component. `ztable` = redshift-grid variant of the same. Other components add their own flag names as approximations land. Confirmed 2026-05-19.
- [ ] Pre-decide Q1 (per_component wins if fused diverges)?
- [x] approx validation policy: `ztable` requires `wave_precomp`; free-z auto-enables `ztable`; unknown flags raise. Confirmed 2026-05-19.
- [ ] Confirm absolute-mag mechanism (re-projection vs separate `rest_filters=`)?
