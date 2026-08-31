# Cross-compile reuse for catalog fits with per-galaxy `Fixed(redshift)`

> Status: **design**, pre-implementation. Companion to the Phase 5
> spectrum LUT doc.

## The problem in one paragraph

A typical photometric-redshift catalog run looks like: 10⁴ galaxies,
each with its own `Fixed(redshift)` from a prior step. Today, each new
`Fixed(z)` value triggers a fresh JIT compile because z is baked into
the model's compile signature (the filter effective wavelengths
depend on it, and the precompute table is built for that z). The
compile cost (~5–10 s warm, ~30–75 s cold) dominates the wall-clock
per galaxy, often by 10×.

The workaround already in main is `WavePrecomp(z_min, z_max)`, which
builds a ztable so a *single* model handles a redshift range
continuously. But that requires the astronomer to pre-decide the
range and explicitly opt in. For a catalog fit at `Fixed(z)` per
galaxy, the natural user-facing shape is: "use my catalog of z values,
do not recompile for each."

## Why this is hard

Three things bake z into the compile signature today:

1. **Filter effective wavelengths** — `λ_eff(z) = λ_eff_obs / (1+z)`
   in rest-frame for the SSP integration. Different z → different
   `filter_eff_waves` → different LUT shape.
2. **Cosmology** — `d_L(z)` enters the `F_ν` projection. Today
   computed eagerly; could be a runtime call.
3. **IGM transmission** — `T_IGM(λ_obs, z)` is evaluated per call.
   Already z-dependent at runtime in the current path.

(1) is the big one. The `ztable` mechanism already exists for (1):
`WavePrecomp(z_min, z_max)` builds a `(n_z, n_filter)` table; at
runtime, the framework interpolates at the current `params["redshift"]`.

So the missing piece isn't building new infrastructure — it's
**re-routing `Fixed(redshift)` through the existing ztable path**
when the astronomer is about to use the same model for N galaxies.

## What the astronomer would write

The end state should look like:

```python
model = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    sfh={"type": "dpl", "all_params": FREE},
    redshift=Fixed(),       # placeholder — value comes from the catalog row
    approx=WavePrecomp(),   # the ztable is automatic
)

for row in catalog:
    posterior = model.fit(
        data=row.fluxes,
        params={"redshift": row.z},   # injected per row, no recompile
    )
```

`Fixed()` with no value would be a sentinel meaning "this parameter
is fixed per-call, supply it via `params=`". `params["redshift"]`
threads through the existing ztable path.

> **2026-09 status note:** `Fixed(DEFAULT)` (the sentinel `DEFAULT`, legal
> only as the argument of `Fixed(...)`) has since shipped and occupies
> the "argument-carrying `Fixed` with a deferred value" space sketched
> above — but it defers to the *registry* default, resolved at parse time,
> not to a per-call runtime value. Bare `Fixed()` (no argument at all, as
> sketched here) remains reserved and unimplemented; it is not
> `Fixed(DEFAULT)` and solves a different problem. Approach B below should
> target the runtime-override mechanism that shipped in the meantime —
> `Fitter(..., params_override={...})` in
> `src/tengri/inference/fitter.py`, which already validates that overridden
> keys are fixed (not free) parameters and threads a `redshift` override
> through the ztable path — rather than building the `overridable=` sketch
> from scratch.

Alternatively, keep `Fixed(z)` (an explicit per-fit value) and let
the model accept an override at call time:

```python
posterior = model.fit(data=row.fluxes, overrides={"redshift": row.z})
```

## What needs to change

### Approach A — minimal: `WavePrecomp` becomes ztable by default

When `Fixed(redshift)` is set in `spec`, the resolver routes it
through the ztable (today's `WavePrecomp(z_min, z_max)` path) anyway,
with `z_min = z_max = Fixed.value`. That doesn't help the catalog
use case but normalizes the path.

Then add an opt-in `WavePrecomp(catalog_z_range=(z_min, z_max))` so
the astronomer says "I'll iterate over a catalog spanning this z
range, build me a ztable that covers it." Per-galaxy fits inject z
through `params["redshift"]`.

This is **half a day of work** — ztable infrastructure exists; we're
adding a constructor knob and a small router change.

### Approach B — full: any Fixed(*) param overridable at call time

Generalize to "any `Fixed(x)` parameter can be lifted to a runtime
input via an explicit override." `model.fit(data, overrides={"x":
new_value})` re-uses the JIT trace because the compile signature for
`x` is "runtime parameter slot", not "baked constant".

This requires teaching the compile-signature machinery that overrides
are pre-declared at construction. Astronomer opts in:

```python
model = SEDModel.build(
    ...,
    redshift=Fixed(...),
    overridable=("redshift",),    # this Fixed is overridable per call
)
```

The compile signature treats `redshift` as a runtime parameter; the
forward pass reads its value from `params`.

This is **a week of work** — touches the compile-signature module,
the Parameters builder, and every inference engine that constructs
the parameter dict for a fit.

## Recommendation

Ship **Approach A first.**

1. It maps directly onto the existing ztable infrastructure.
2. The catalog-fit case (the main motivator) is satisfied.
3. Approach B is the generalization — defer until a second use case
   (e.g. per-galaxy `Fixed(distance_modulus)`) demands it.

## File-by-file changes for Approach A

| Path                                                | Change                                                |
|-----------------------------------------------------|--------------------------------------------------------|
| `src/tengri/forward/sed_model.py:WavePrecomp`       | Add optional `catalog_z_range: tuple[float, float] | None = None` |
| `src/tengri/forward/sed_model.py` resolver          | When `catalog_z_range` is set, build the ztable; when `Fixed(z)` is set on spec, route through the ztable with `z_min = z_max = z` (no behavior change for the simple case) |
| `src/tengri/components/stellar/component.py`        | Already supports ztable — confirm nothing breaks      |
| `src/tengri/observation/predict_via_precomp.py`     | Confirm cosmology + IGM read z at runtime, not from compile signature |
| `tests/contract/test_cross_compile_catalog_fit.py`  | NEW — same model fit at three different z values produces three results with **one** JIT compile |
| `bench/scripts/benchmark_catalog_fit.py`            | NEW — time a 100-galaxy catalog fit before/after      |
| `docs/dev/archive/forward-model-architecture.md`            | New §6.5 "Catalog fits at per-galaxy Fixed(z)"        |

## Acceptance

* **No-regression**: existing `Fixed(z)` single-galaxy fits remain
  bit-identical to current main.
* **Single compile**: a synthetic 10-galaxy catalog fit triggers
  exactly one JIT trace (verified by capturing the compile cache
  hit/miss counts).
* **Posterior parity**: per-galaxy posterior at `Fixed(z=0.5)` agrees
  with a single-galaxy run with `Fixed(z=0.5)` to bit precision (the
  ztable is interpolated at exactly the grid point so it should
  match).

## Out of scope for v0

* Approach B (general `Fixed → runtime` lift).
* Free-`redshift` catalog fits — they already work via the ztable.
  This doc is about the `Fixed(z)` case.

## Open design questions

1. **Constructor knob name**: `catalog_z_range=(z_min, z_max)` vs
   `z_range=(...)` vs let the astronomer set `n_z`/`z_min`/`z_max`
   directly on `WavePrecomp`. The existing
   `WavePrecomp(n_z=200, z_min=0.0, z_max=3.0)` already provides the
   low-level shape; the catalog-fit knob is a documentation-level
   alias that nudges the astronomer to think about catalog coverage.
2. **`params["redshift"]` shape**: scalar (per-call) or array (vmap
   over the catalog in one trace). The first cut is scalar per call;
   a `vmap` version is a follow-up that integrates with the inference
   layer's batched-likelihood support.
3. **Cosmology lookup**: confirm `luminosity_distance(z)` is on the
   exact path (not closure-baked into the JIT trace). If yes, no
   change needed. If no, lift the cosmology call to runtime.

## References

* `src/tengri/forward/sed_model.py:WavePrecomp` (current dataclass)
* `WavePrecomp(z_min, z_max)` ztable path documented in
  `docs/dev/archive/forward-model-architecture.md` §5
* `docs/dev/orchestrator_ssp_threading.md` — earlier Phase 4 work on
  threading arrays as JIT runtime inputs (analogous mechanism)
