# Orchestrator SSP threading — prerequisite for kernel deletion

**Status:** open design problem; blocking kernel-adapter deletion (Phase 3d
PR-2 / Phase 4).
**Authored:** 2026-05-20 (after Phase 3d-3, while planning kernel removal).

## What this is

The legacy kernel adapter family (`src/tengri/forward/_kernels/`) survives
in `main` after Phase 3d because removing it regresses a load-bearing XLA
performance optimization. This doc names the optimization, the problem it
solves, and the work needed before the kernels can safely be deleted.

## Why we have a `mode="traced"` back-door today

`_predict_photometry_traced` is documented as "un-JIT'd path for use
inside JAX tracing (NIFTy VI)". It does two things that are easy to
conflate:

1. **JIT-in-JIT avoidance.** When the forward model is called from inside
   an outer `@jax.jit`'d function (an inference loss, NIFTy's
   `signal_response`), the *inner* compositional kernel's own
   `@jax.jit` decoration is wasteful — we want the body inlined into the
   outer trace. `_photometry_raw` is the same function without the
   decorator.

2. **HLO size optimization via SSP threading.** Reads
   `src/tengri/forward/sed_model.py:4482` and friends:

   ```python
   return raw(
       sfr_on_ssp,
       params,
       ssp_flux_traced=self.ssp_data.ssp_flux,
       ssp_lgmet_traced=self.ssp_data.ssp_lgmet,
   )
   ```

   The SSP grid arrays are threaded as JIT-traced *kwargs* rather than
   captured via closure. XLA treats them as runtime inputs instead of
   baking them into the HLO as `xla::Constant` ops. On the MIST grid
   that's a **114 MB constant per compile** — slow compile, big HLO,
   high peak memory.

The first concern (JIT-in-JIT) is already covered by `observation.predict`,
which is *not* internally `@jax.jit`'d. The second concern is **not**
covered by the new path.

## The cross-galaxy cache works either way

`compile_signature()` already includes `ssp_lgmet_id` (hash of the SSP
metallicity grid values) and `filter_trans_id` (hash of the transmission
curves). Two `SEDModel` instances pointing at the same SSP file with the
same physics share one compile via the structural kernel cache. That
machinery is independent of whether the SSP grid is closure-captured or
threaded — it just gates **whether** a compile is reused, not **what** the
compile contains.

So cross-galaxy sharing per-SSP is safe today. The HLO size, compile time,
and peak memory are not.

## What the new path looks like today

`predict_observables_jit` (line 4083) wraps `observation.predict(state,
params, ...)` in a `@jax.jit`'d closure. The closure body is:

```python
def _impl(params):
    state = self.predict_state(params)       # ← orchestrator
    full = {**fixed_values, **params}
    return observation.predict(state, full, ..., observables_type=...)
```

Only `params` is threaded. `self.ssp_data` is captured via the
``predict_state`` ``self`` reference; every component in the chain reads
``self.ssp_data.ssp_flux`` / ``self.ssp_data.ssp_lgmet`` through its own
closure capture. Under XLA, both arrays land in the compiled HLO as
``Constant`` ops.

## What needs to change

Every `SEDComponent.apply(state, params)` that touches SSP arrays must
accept them as *parameters*, not via closure capture. Two natural shapes:

**Option A — thread on the state.** Put `ssp_flux`, `ssp_lgmet`, and any
other ssp_data fields on the `ForwardState` produced at the head of the
chain (e.g. by a synthetic root component or by the orchestrator entry
point). Subsequent components read them off `state` instead of `self`.

  + No signature change for `apply()`.
  + Natural fit for the existing `state.derived` contract.
  − `state` grows by ~150 MB worth of array references (cheap, but does
    pollute the readable state surface).

**Option B — thread via a side `ssp` parameter.** Change the orchestrator
to call `apply(state, params, ssp=ssp_data)` for every component, and
have `SEDComponent.apply` accept a third positional / keyword argument.

  + Keeps `state` focused on derived science quantities.
  − Signature change for every component (~15 files).

Either way the public `predict_observables_jit` closure needs to thread
`ssp_data.ssp_flux` and `ssp_data.ssp_lgmet` as `jax.jit` arguments. The
existing `compile_signature()` keying gates correctness; threading gates
performance.

## Phase 4-A landed (2026-05-20): runtime-parameter threading

PR #135 includes a smaller cousin of the SSP-threading work — threading
**fixed parameter values** as a runtime JIT input instead of closure
capture. This unblocks cross-galaxy compile reuse when galaxies differ
on a fixed parameter whose value is read at *runtime* (e.g. ``dust_tau_bc``,
``met_logzsol``, ``neb_logU``) — but **not** for fixed values used at
*chain-construction time* (notably ``redshift`` when the stellar LUT is
built at the spec's fixed z).

Concrete changes:

1. ``compile_signature()``'s ``spec_fixed_id`` now keys on fixed-parameter
   *names* only (sorted tuple), not name+value pairs.
2. ``predict_state(params, fixed_values=None)`` accepts an optional
   ``fixed_values`` override; when ``None`` it falls back to
   ``self.spec.get_fixed_values()``.
3. ``_get_or_build_predict_observables_jit`` builds a closure of shape
   ``_impl(params, fixed_values)`` — both are JIT runtime inputs.
4. ``predict_observables_jit(params)`` threads
   ``self.spec.get_fixed_values()`` through explicitly:
   ``fn(params, self.spec.get_fixed_values())``.

Two contract tests (in ``tests/unit/test_approx_kwarg_contract.py``)
pin the behaviour: same compile_signature across galaxies, correct
per-galaxy result, agreement with the non-JIT path.

**What this does NOT solve** (still open, the rest of this doc):

- Per-galaxy ``Fixed(redshift)`` catalogs — the stellar LUT is built
  at the spec's fixed z, so each galaxy has a structurally different
  chain. The workaround is to build with ``approx=WavePrecomp(z_min, z_max)``
  covering the catalogue range and pass the per-galaxy z via ``params``
  on a *free-redshift* spec.
- SSP-as-HLO-constant: the 114 MB SSP grid is still closure-captured
  via ``self.ssp_data`` inside the chain's components.

Both remain Phase 4-B / future work.

## Related catalog-inference concerns (Phase 3d-4, 2026-05-20)

PR #135 introduced ``approx=WavePrecomp(...)`` as the build-time opt-in
for the LUT path, but discovered two additional gaps when validating the
many-galaxy story:

### 1. Cache collision when WavePrecomp config differs (fixed)

``compile_signature()`` originally only included the boolean
``wave_precomp`` / ``ztable`` / ``igm`` flags from ``self._approx``. Two
models built with ``WavePrecomp(n_z=100)`` and ``WavePrecomp(n_z=200)``
shared a signature → the structural kernel cache reused the first
model's compiled LUT for the second, silently miscompiling the second
galaxy. Fixed in this PR by adding the resolved ``n_z``, ``z_min``,
``z_max`` to the signature alongside the boolean flags.

### 2. `predict_observables_jit` cannot route through the LUT (open)

The non-JIT ``predict_observables`` and ``predict_photometry`` route
through ``observation.predict_via_precomp`` when ``WavePrecomp`` is set
— callers see the LUT speedup. But ``predict_observables_jit`` (the
catalog-fitter entry point) cannot, because ``predict_via_precomp``
carries Python-level guards that don't trace under ``jax.jit``:

```python
# observation.py:804
dust_active = l_ir is not None and float(l_ir) > 0.0    # ← concretizes l_ir
# observation.py:827
nebular_additive_active = sed_nebular is not None and float(jnp.max(jnp.abs(...))) > 0.0
```

These guards exist to give a clear ``NotImplementedError`` if the user
calls ``predict_via_precomp`` on a model whose dust / nebular component
ran on the wave grid but didn't publish a precompute. They're *structural*
checks (does this combination of components support the LUT?) but
implemented as *runtime* checks (with ``float()`` calls).

**The fix** is to lift these guards to build-time validation — a
``_validate_lut_path_or_raise()`` that runs once when ``approx=WavePrecomp``
is requested, checks the published derived-key set against the resolved
component chain, and raises immediately if the combination is unsupported.
After that, the call-time path is free of ``float()`` concretizations
and can be ``jax.jit``'d.

Until then, catalog inference doesn't see the LUT speedup. Direct callers
of ``predict_photometry`` and ``predict_observables`` do.

## Phase 4-C landed (2026-05-21): nebular backend threading

Extended Phase 4-B's SSP-threading pattern to the nebular backends that
close over large arrays at ``apply()`` time:

- ``CueBackend.weights`` (~10–20 MB of NN layer weights)
- ``CloudyGridBackend.grid`` (~5–30 MB interpolation tables)

These were the two currently selectable via ``NebularConfig.backend``.
``SEDModel._template_data_for_jit()`` duck-types ``.weights`` / ``.grid``
on the nebular backend object, so future-wired backends with the same
attribute name participate automatically.

## When does a closure need threading? The rule

A large array bakes into HLO as a `Constant` op **whenever it's read
inside `apply()` under JIT through any path that JAX treats as
closure-captured** — instance attributes (`self.X`), module-level
lazy-loaded caches, or factory-returned closures that hold the array
in their captured scope all qualify. Threading is the workaround:
pass the array as an explicit runtime input so JAX makes it a
`Parameter` op instead.

The accurate rule is:

| Pattern | Examples | Threading needed? |
|---|---|---|
| Backend `self.weights` / `self.grid` read directly | Cue, CloudyGrid | **yes** |
| Module-level `@functools.cache` returning a closure that holds the grid as a `jnp.array` (via `ensure_compile_time_eval`) | SKIRTOR, DL07, DL14, Dale 2014, Astrodust, BOSA, Draine 2021 PAH | **yes** — same HLO impact, just a different path |
| Per-component pytree `_state` (frozen dataclass with `jnp.ndarray` fields, the state IS a JIT input) | GRAHSP via `GRAHSPSEDComponentState` | no |
| Small (< 1 MB) | IGM Inoue 2014 tables | no — negligible Constant cost |

**Correction (2026-05-21):** an earlier version of this doc claimed
"dust IR and AGN templates DON'T need threading because they're
preintegrated at precompute() time." That's wrong for most of them.
The legacy `dust_emission_precompute.py` and `skirtor_precompute.py`
paths were used by the deleted kernel adapter family (Phase 6); the
**current** ``DustSEDComponent`` and ``AGNSEDComponent.apply()``
paths reach templates via the module-level lazy-loaded registry, so
the templates **do** close over the JIT trace today. Only GRAHSP is
genuinely safe (its templates live on a per-component pytree state).

That correction means the threading work is **not finished** for
dust IR and AGN SKIRTOR. Both are tracked in
[#138](https://github.com/suchethac/tengri/issues/138).

## Known unwired backends (future work — tracked in [#138](https://github.com/suchethac/tengri/issues/138))

The following backend files exist but aren't currently dispatched by
``NebularSEDComponent.apply()`` — they're standalone modules, fallback
wrappers, or AGN-NLR backends without a forward-path consumer:

- ``components/nebular/cloudy_cb19.py:CB19Backend`` — wrapped by
  ``NebularContinuumFallback``; user-facing via wrapper, not a
  ``config.backend`` value.
- ``components/nebular/mappings_photo.py:MappingsPhotoBackend`` and
  ``MappingsAGNBackend`` — exported but no forward consumer.
- ``components/nebular/agn_nebular.py:SynthesizerNLRBackend`` and
  ``FeltreNLRBackend`` — exported but no forward consumer.

When these are wired into ``NebularSEDComponent.apply()`` dispatch (or
into an analogous AGN-nebular component), the ``_template_data_for_jit``
duck-typing already covers them — no further code change required here.
File the wiring work as a follow-up issue.

## What this blocks

Until the orchestrator threads SSP as a traced kwarg:

* **Cannot delete** `src/tengri/forward/_kernels/_adapters.py`,
  `compositional.py`, `hybrid.py` — they implement the optimization the
  new path doesn't yet have.
* **Cannot delete** `_predict_photometry_traced`,
  `_predict_spectrum_traced` — inference internals (~100 sites) use
  `mode="traced"` for the same optimization.
* **Cannot drop** the user-facing `mode="compositional"|"hybrid"|"exact"`
  values from `predict_photometry`/`predict_spectrum` — they survive
  alongside the kernels.

## Acceptance criteria for the future PR

1. `predict_observables_jit`'s HLO contains the SSP grid as `Parameter`
   ops, not `Constant` ops. Verify via `jax.xla_computation(...).as_hlo_text()`
   on a representative compile.
2. Compile time + peak memory benchmark on the MIST grid (114 MB SSP)
   regresses by < 10 % vs. the current `mode="traced"` path.
3. Cross-galaxy compile cache still hits when two models share
   `compile_signature()` — same SSP file, different galaxies.
4. Bit-exact agreement with the current compositional kernel output on
   a fixed-z stellar-only photometry test.
5. After the threading lands, the kernel adapter family + the legacy
   `mode=` cascade can be deleted in the same PR (covered by the existing
   Phase 3d temp plan).

## Pointers

* Today's threading site:
  `src/tengri/forward/sed_model.py:4451-4486` (`_predict_photometry_traced`)
* Today's compile cache:
  `src/tengri/forward/sed_model.py:2540` (`compile_signature`),
  `src/tengri/inference/_model_cache.py` (`ModelCacheOwner`).
* Today's untraced HLO path:
  `src/tengri/forward/sed_model.py:4112-4126`
  (`_get_or_build_predict_observables_jit._impl`).
* The kernel deletion plan: `docs/dev/phase3d_temp_plan.md`.
