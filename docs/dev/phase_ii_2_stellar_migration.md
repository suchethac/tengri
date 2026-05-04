# Phase II-2 — Stellar SEDComponent migration

> Status: **II-2.0 verified done; II-2.1 → II-2.5 implemented
> (2026-05-03; II-2.5 partial — dpl yes, non-parametric SFHs deferred);
> II-2.6 unblocked at the machinery level (PipelineState pytree fix);
> public-API integration into Galaxy.predict() open.** This document scopes the migration as a series
> of focused PRs. See `~/.claude/plans/i-want-you-to-agile-matsumoto.md`
> for the agreed entropy budget enforced on every PR.

## Why this is the hard one

`StellarSEDComponent` is the largest of the seven Phase II adapters
because it absorbs three currently-separate concerns:

1. **SFH** — `tengri.components.sfh/` (parametric models + GP field +
   chemical evolution + non-parametric forms)
2. **SPS** — `tengri.components.sps/` (DSPS wrapper + SSP grid loading
   + CSP weight computation + alpha-element interpolation)
3. **Chemistry / metallicity** — currently spread across both
   subpackages (`MET_REGISTRY`, `closed_box_metallicity`,
   `tabulated_metallicity_on_ssp_grid`, `psb_two_step_metallicity`,
   ...)

It is also the *upstream* of every other adapter. Dust attenuation
needs per-age luminosities; nebular needs ionising photon production;
radio/X-ray need SFR + M_*. So the stellar component is what unblocks
richer versions of every downstream adapter.

## What this component owns (declared parameters)

The stellar component declares **all** `sfh_*`, `met_*`, and `chem_*`
parameters. The exact list depends on `config.sfh_model` /
`config.metallicity_model`:

```python
@dataclass(frozen=True)
class StellarSEDComponentConfig(SEDComponentConfig):
    name: str = "stellar"
    sfh_model: str = "tsnorm"      # registered name from SFH_REGISTRY
    field: bool = True              # add stochastic GP field on top
    n_grid: int = 64                # log-time grid resolution
    metallicity_model: str = "delta"  # "delta" | "ramp" | "chem_evol" | "tabulated"
    sps_backend: str = "dsps"       # "dsps" | "dsps_native"
    use_alpha_grid: bool = False    # alpha/Fe SSP interpolation
```

Representative declared-parameter sets per `sfh_model`:

| `sfh_model` | Parameters declared |
|---|---|
| `tsnorm` (truncated skew-normal) | `sfh_tsnorm_log_peak_sfr`, `sfh_tsnorm_peak_lbt_gyr`, `sfh_tsnorm_width_gyr`, `sfh_tsnorm_skew`, `sfh_tsnorm_trunc` |
| `dpl` (double power-law) | `sfh_dpl_alpha`, `sfh_dpl_beta`, `sfh_dpl_tau`, `sfh_dpl_norm` |
| `field` (PSD-governed GP, used as additive on top of any mean) | `sfh_field_psd_sigma`, `sfh_field_psd_tau_myr`, `sfh_field_xi` (an n_grid-dim vector of unit-variance Gaussian draws) |
| `continuity` (binned non-parametric) | `sfh_continuity_logsfr_ratios` (vector of length `n_bins-1`) |
| `dirichlet` | `sfh_dirichlet_logsfr_ratios` (vector) |

Metallicity additions:

| `metallicity_model` | Parameters declared |
|---|---|
| `delta` | `met_logzsol` |
| `ramp` | `met_logzsol_initial`, `met_logzsol_final` |
| `chem_evol` (closed-box) | `chem_zform`, `chem_yield` (or single `met_logzsol_anchor` depending on chosen variant) |
| `tabulated` | none — table baked into `config` |

Plus optional `met_alpha_fe` when `use_alpha_grid=True`.

## What this component reads from upstream

**Nothing.** Stellar is the head of the pipeline — the `PipelineState`
arrives carrying only `wave` and (optionally) the bare `redshift`
parameter. Stellar produces the first non-trivial SED.

## What this component publishes to PipelineState

This is the **contract** every downstream adapter relies on. Once
fixed, changing it requires bumping a major version.

| Slot | Type | Units | Meaning |
|---|---|---|---|
| `state.sed_intrinsic` | `jnp.ndarray, shape (n_wave,)` | erg/s/Hz (rest-frame L_nu) | The pre-attenuation, pre-emission stellar SED, normalised to a stellar mass of `10**log_mstar` |
| `state.derived["log_mstar"]` | scalar `jnp.ndarray` | log10(M_⊙) | Stellar mass formed (or surviving — see "open question" below) |
| `state.derived["sfr"]` | scalar `jnp.ndarray` | M_⊙/yr | Current-time star-formation rate (i.e. SFH evaluated at lookback time = 0) |
| `state.derived["sfr_10myr"]` | scalar `jnp.ndarray` | M_⊙/yr | SFR averaged over the last 10 Myr (radio + X-ray prefer this over the instantaneous value) |
| `state.derived["sfr_100myr"]` | scalar `jnp.ndarray` | M_⊙/yr | SFR averaged over the last 100 Myr |
| `state.derived["L_age"]` | `jnp.ndarray, shape (n_age,)` | erg/s | Age-resolved bolometric luminosity, on the SSP age grid; consumed by two-component dust attenuation |
| `state.derived["lnu_age"]` | `jnp.ndarray, shape (n_age, n_wave)` | erg/s/Hz | Age-resolved L_nu *before* CSP weighting, retained for dust components that need per-age attenuation |
| `state.derived["nion"]` | scalar `jnp.ndarray` | photons/s | Ionising photon production rate (λ < 912 Å), consumed by nebular |
| `state.derived["sfh_grid_lbt_yr"]` | `jnp.ndarray, shape (n_grid,)` | yr (lookback time) | The SFH evaluation grid — useful for diagnostics and for components doing time-resolved analysis |
| `state.derived["sfr_history"]` | `jnp.ndarray, shape (n_grid,)` | M_⊙/yr | The realized SFH on `sfh_grid_lbt_yr` |
| `state.derived["log_metallicity_history"]` | `jnp.ndarray, shape (n_grid,)` | log10(Z) absolute | Per-time-bin metallicity, for diagnostics + chem_evol mode |

### Open question: surviving vs formed mass

`log_mstar` is ambiguous. Two reasonable choices:

1. **Mass formed** — integral of SFR over cosmic time. Easier to compute; preferred by SFH-side code.
2. **Surviving mass** — formed × DSPS return-fraction at the galaxy age. What observers measure photometrically.

Resolution path: pick one for the contract (recommend **surviving mass** since it matches what radio/X-ray scaling laws were calibrated against), publish *both* in `state.derived` (`log_mstar` = surviving, `log_mstar_formed` = formed), document the choice in `core/component.py` next to `BARE_NAME_ALLOWLIST`.

## What other Phase II adapters need from this contract

Validated against the four adapters already migrated:

- **DustAttenuationSEDComponent (single-screen)** — needs `state.sed_intrinsic`. ✓
- **DustTwoComponentSEDComponent** (not yet built) — needs `state.derived["lnu_age"]` and `state.derived["sfh_grid_lbt_yr"]` to compute the birth-cloud mask in age space.
- **NebularSEDComponent (Cue)** — needs `state.derived["nion"]` (or equivalent ionising-photon production), `state.derived["sfr"]` (for the current-time U parameter), `params["met_logzsol"]` directly via prefix slicing.
- **AGNSEDComponent** — independent of stellar at the SED level (publishes `L_agn_bol` itself), but its host-related radio/X-ray pieces need `state.derived["log_mstar"]`. ✓
- **RadioSEDComponent** — already reads `state.derived["log_mstar"]`. ✓
- **XRaySEDComponent** — already reads `state.derived["sfr"]` and `state.derived["stellar_mass"]`. (Note: today XRay reads `stellar_mass` not `log_mstar` — Phase II-2 should standardise on `log_mstar` everywhere and have XRay convert via `10 ** log_mstar`.)
- **IGMSEDComponent** — independent. ✓

**Action item from this audit:** rename `state.derived["stellar_mass"]` (used by XRaySEDComponent) → `state.derived["log_mstar"]` (with a `10 **` exponentiation inside the X-ray adapter). One-liner change; do it before Phase II-2 starts so the contract is consistent.

## Migration sequence (target: one PR per row)

| PR | Scope | Lines moved | Tests |
|---|---|---|---|
| **II-2.0** ✅ | Standardise on `log_mstar` in XRay. **Verified done 2026-05-03**: `components/xray/component.py:171` already reads `state.derived.get("log_mstar", 10.0)`. No code change needed. | 0 | XRay integration tests already cover both |
| **II-2.1** ✅ | New `src/tengri/components/stellar/` package. Move (not rewrite) `sfh/` and `sps/` into it. Re-export shims keep `tengri.components.sfh.*` and `tengri.components.sps.*` working with `DeprecationWarning`. **Implemented 2026-05-03** via `sys.modules` aliasing in two shim `__init__.py` files; all 26 internal `src/tengri/` importers updated to canonical paths. | ~3500 (move) | full unit + integration suite still passes |
| **II-2.2** ✅ | `StellarSEDComponent` adapter for the most common case: `tsnorm` SFH + `delta` metallicity + `dsps` backend. Hardcoded `field=False` for now. Lives alongside the existing `sed_model.py` tier-dispatch — does NOT replace it. **Implemented 2026-05-03**: full 11-key contract published, lints clean, smoke-tested at z=0 on PRSC-MILES SSP. Architectural decision: `ssp_data` is a constructor field on the component (consistent with Radio/IGM/XRay holding their config); `precompute()` stays a no-op marker. Deferred: bit-exact rtol=1e-8 equivalence test vs legacy + JIT compatibility (PipelineState needs pytree registration; orthogonal). | ~400 | smoke test asserts all 11 derived keys finite + sensible magnitudes |
| **II-2.3** | Add the field branch (PSD-governed GP). Adds `sfh_field_*` parameters and the `state.derived["sfr_history"]` publication. | ~400 | parametric test over `(field=False, field=True)` |
| **II-2.4** | Add `ramp` and `chem_evol` metallicity modes. | ~300 | parametric test over the 4 metallicity modes |
| **II-2.5** | Add `dpl`, `continuity`, `dirichlet`, `dense_basis` SFH modes. | ~500 | parametric test over each registered SFH model |
| **II-2.6** | Wire the orchestrator into `Galaxy.predict()` for the cases where stellar+dust+igm+radio+xray are sufficient. Tier-dispatch path keeps running for nebular+AGN until those land. | ~200 | end-to-end fit on a quiescent galaxy (no nebular, no AGN) using the new orchestrator |

Total: ~5500 lines moved or added, target 6 PRs, ~1 week per PR.

## Verification gates (every PR must pass)

1. **Lint clean** — `ruff check src/ tests/ tools/` passes.
2. **Numerical equivalence** — for every `(sfh_model, metallicity_model)` pair migrated, the new adapter produces an SED within `1e-10` relative tolerance of the legacy `predict_*` path on a 100-galaxy parameter sweep. Recorded in `tests/integration/test_stellar_pipeline.py`.
3. **Posterior equivalence** — pick one mock galaxy, run inference via both paths (legacy `Fitter` vs new orchestrator), assert posterior medians agree to 0.05σ in every dimension. This is the strongest test and is the gate before II-2.6 ships.
4. **Compile-cache compatibility** — `geoVI` cold compile ~75s reproducible; warm compile via the cache restores in <1s. The persistent JAX cache must not need clearing between legacy and new paths.
5. **Cross-validation** — `pytest -m crossval tests/crossval/` passes within tolerance against bagpipes/FSPS.
6. **Benchmark parity** — `JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_forward_model.py` warm time within 5% of pre-migration baseline.
7. **Backwards compatibility** — the existing `Galaxy.from_arrays(...)` and `Fitter(...).run(...)` user-facing APIs continue to work unchanged; the new orchestrator is reachable via `model.predict(..., backend="component")` (or similar opt-in flag) until II-2.6 flips the default.

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Stochastic GP field path produces different posterior between legacy `vi` (NIFTy) and new `vi_native` (pure JAX) — already known per CLAUDE.md gotcha — and the migration accidentally locks one of them in | Medium | Keep both `vi` and `vi_native` reachable through II-2.6; add a regression test asserting both produce posteriors that agree on `sfh_field_psd_sigma` to within 2σ on a known mock |
| Surviving-vs-formed mass choice breaks calibration of `xray_xrb`, `radio_sfr_*` | High if not addressed | Resolve in II-2.0 BEFORE the stellar adapter lands — pick surviving-mass for `log_mstar`, document, update XRay/Radio adapters consistently |
| The `_kernels/compositional.py` BUG-NSS-02 ramp-mode work currently in-flight conflicts with II-2.4 | Medium | Land the BUG-NSS-02 fix first, rebase II-2.4 on top; the new component path can pick up the same fix. Coordinate with whoever owns the in-flight branch |
| Parametric SFHs use yr/Gyr/Myr unit conventions inconsistently in legacy code (`psd_tau_myr` Myr, `tau_sfh` yr, `tau_peak_gyr` Gyr) — the migration accidentally introduces a 10⁶ off-by-million | High in any unit-shifting code | Add a unit-contract regression test in II-2.1 that asserts every parametric SFH produces the same numerical SFH array on a fixed param set in legacy and new paths |
| Mass loss return-fraction interpolation in DSPS is computed slightly differently in `dsps_native` vs `dsps` | Low (already a known divergence between the backends) | Pick one backend as canonical for the new component (recommend `dsps`) and gate `dsps_native` behind `config.sps_backend="dsps_native"` |

## Out of scope for Phase II-2

- Inference engine changes — `Fitter` keeps its tier-dispatch logic until Phase II-5.
- Likelihood implementations — current `_run_*` methods stay where they are.
- Removing legacy code — every legacy code path stays runnable through deprecation; II-2.6 flips the default to the new path but does not delete the old one.
- Splitting `forward/sed_model.py` — separate refactor (REFACTOR.md item).
- Renaming free parameters — every `sfh_*`/`met_*` name is unchanged from today's code.

## Definition of done

Phase II-2 is complete when:

1. All six PRs above have landed.
2. The orchestrator handles a galaxy with stellar + dust + igm + radio + X-ray and produces an SED bit-identical to the legacy `predict_*` path within `1e-10` relative tolerance.
3. A full mock recovery run (`scripts/test_*` family) using the new orchestrator path produces posteriors statistically indistinguishable from the legacy path on a 50-galaxy sample (Wasserstein distance < 0.1σ in every dimension).
4. `forward/sed_model.py` has gained a `use_component_orchestrator` flag, defaulting to `True`, and the legacy code path is reachable via `False` for one minor version of fallback safety net.

After II-2.6 ships, Phase II-3 (Dust two-component + Nebular) can begin.

## Open questions — resolved 2026-05-03

1. **`log_mstar` = surviving stellar mass.** Reason: it matches what
   observers actually constrain (surviving M_* is what a photometric
   SED ratio is sensitive to, formed mass is a derived quantity from
   IMF+age). Formed mass is still published as `log_mstar_formed` for
   diagnostic continuity with prior Tengri results.

2. **`lnu_age` is eagerly published** as
   `state.derived["lnu_age"]`, shape `(n_age, n_wave)`. ~3 MB at
   `n_age=140, n_wave=2700`. Reason: simpler contract; downstream
   adapters (dust two-component, nebular Cue) need it predictably; the
   memory cost is irrelevant at typical sample counts. A "request"
   channel was rejected as premature optimisation.

3. **`StellarSEDComponent.parameter_prefix` is a tuple** —
   `("sfh_", "met_", "chem_")` — and the orchestrator's prefix-slicer
   accepts both `str` and `tuple[str, ...]`. Reason: stellar physics is
   irreducibly coupled (SFH × metallicity × IMF+chem-evol drive a
   single SED). Splitting into `ChemSEDComponent` would create
   cross-component reads with no physical separation. The tuple is a
   one-line orchestrator change with full backward compatibility.

4. **Numeric equivalence target: `rtol=1e-8`.** Reason: SSP grid
   interpolation accumulates floating-point error past `1e-10`. The
   `1e-10` bar from IGM/radio/X-ray is achievable because those are
   closed-form algebra; stellar passes through DSPS / SSP grid
   interpolation where `1e-8` is the natural precision floor.

These resolutions are baked into the Phase II-2 skeleton at
`src/tengri/components/stellar/component.py` (added 2026-05-03).
