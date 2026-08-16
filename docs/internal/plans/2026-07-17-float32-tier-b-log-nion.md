# Tier B step 1: `log_nion` contract — log-domain Q_H across the forward model

## Context

Tier A (PR #1193, merged as `5769a3e46`) made the forward model mixed-precision float32-safe by
reparametrizing multiplicative scale seams as log10 offsets (`utils/scale.py`). Issue #1206 tracks
Tier B (pure float32, JAX-Metal — no f64 fallback); its **recommended first task** is the
`log_nion` contract: the ionizing-photon rate Q_H ≈ 1e56 photons/s overflows float32 max (3.4e38)
no matter how the summands are computed, so the *output contract* must become logarithmic.

The user's acceptance bar (carried over from Tier A): **f64 behavior preserved to rtol ≤ 1e-12**
vs pre-change main, pure-f32 correctness verified under `jax.enable_x64(False)` (explicit
`.astype(f32)` under x64 is NOT valid — intermediates upcast and mask failures), subagent-driven
TDD execution, ship as a PR referencing #1206.

### Verified failure chain under pure f32 (all file:line checked in-tree)

1. `_integrate_nion` (`components/stellar/component.py:494-536`): `integrand = sed_lnu/(H_PLANCK·ν)`
   adds ~+10.7 decades (÷2.2e-11) → ~1e47 → **inf**; the trapezoid result ~1e56 → inf; the #537
   boundary correction then computes `inf − inf = nan`. (Observed: `q_h = nan`.)
2. **Cue high-level path** `_derive_cue_params_from_ssp` (`cue.py:~1334-1480`) — this is the branch
   the standard model actually takes (stellar always publishes `age_weights`, so the
   `derived["nion"]` fallback at `nebular/component.py:546` never fires):
   - `qh_per_bin = 10.0**logqion_all` (L1406) — logqion up to ~48-52 → inf; the #1001 defense
     `jnp.where(jnp.isfinite(...), ..., 0.0)` then **zeroes every young bin** → `total_qh = 0` →
     `gas_logqion = -99` → **silently zero nebular emission** (a fail-open guard — the bug class
     the user's standing rules forbid).
   - `seg_per_bin = 10.0**log_seglum_all` (L1448) — same overflow→zeroing; and
     `jnp.maximum(seg_tot, 1e-300)` (L1454) — 1e-300 underflows to 0.0 in f32, so the floor is a
     no-op and `logLratio_eff = diff(log10(0))` → -inf/nan.
3. Downstream linear consumers of `derived["nion"]` (complete, verified by grep): the Cue fallback
   (nebular/component.py:546-548, takes log10 immediately), grid-precomp reconstruct
   (nebular/component.py:735-739 → `nebular_grid_precompute.py` `reconstruct_nebular_phot`:
   `nion * 10**log_lpq`), line-precomp runtime (`line_precompute.py:78-81,222`: `nion * lpq`),
   radio thermal (`radio/component.py:566,582` → `sed_quantities.py:712`: `5.5e-28 * q_h`),
   `_q_h_fn` (stellar/component.py:2699-2703), `_xi_ion_fn` (stellar/component.py:2706-2722), and
   `state_to_ionizing_quantities` (`forward/component_factory.py:903-928`). In every linear case
   nion multiplies a tiny per-Q_H factor whose **product is f32-representable** — the same
   multiplicative-seam shape Tier A already solved.

**Out of scope (stated explicitly):** `line_lums` in erg/s (~1e41, `DerivedKey` at
nebular/component.py:254) and the linear `q_h`/emission-line **properties** stay
f32-unrepresentable — that is #1206 item 3 (breaking unit change, sequenced last). CloudyGrid /
mappings_photo derive their own internal per-age Q_H and do not read `derived["nion"]` — deferred,
noted in the boundary doc. So the current xfail test C cannot simply flip; it gets restructured
(Task 6) into a passing log-domain test + a strict xfail for the item-3 remainder.

## Design

### 1. Log-domain core integral (single source of truth)

New module-level function in `components/stellar/component.py`, replacing the body of the current
`_integrate_nion` (which becomes a thin wrapper — they can never drift):

```python
def _integrate_nion_log10(sed_lnu, wave, log10_scale=0.0):
    """log10(Q_H) [dex re photons/s]; no out-of-range intermediate at any step."""
    peak = jnp.max(jnp.abs(sed_lnu))
    peak = jnp.where(peak > 0, peak, jnp.ones_like(peak))
    ell = sed_lnu / peak                      # O(1) normalized L_nu
    nu = C_AA / wave
    nu_edge = C_AA / _HI_LIMIT_AA
    integrand = ell / nu                      # O(1e-16): NO H_PLANCK division here
    ionizing_mask = wave < _HI_LIMIT_AA
    integrand_masked = jnp.where(ionizing_mask, integrand, 0.0)
    idx_below = jnp.argmax(jnp.where(ionizing_mask, jnp.arange(wave.shape[0]), -1))
    integrand_below = integrand[idx_below]
    triangle = 0.5 * integrand_below * jnp.abs(nu[idx_below] - nu[idx_below + 1])
    rectangle = integrand_below * jnp.abs(nu[idx_below] - nu_edge)
    bulk = jnp.abs(jnp.trapezoid(integrand_masked, nu))
    norm = bulk - triangle + rectangle        # #537 correction on O(1) values, BEFORE the log
    pos = norm > 0
    safe = jnp.where(pos, norm, 1.0)          # where-dummy pattern (grad-safe log)
    log10_norm = jnp.where(pos, jnp.log10(safe), -jnp.inf)
    return log10_norm + jnp.log10(peak) - jnp.log10(H_PLANCK) + log10_scale

def _integrate_nion(sed_lnu, wave):
    return pow10(_integrate_nion_log10(sed_lnu, wave))   # exp(-inf)=0 → zero-flux exact
```

Key points (each verified): the peak factoring happens **before** the `/(hν)` division (the first
overflow); H_PLANCK's −26.2 decades are folded into the log sum; the #537 boundary correction is a
signed combination of O(1) terms taken before the single `log10`; zero flux maps to `-inf` in log
and **exactly 0.0** through `pow10` (matching today), with the where-dummy keeping gradients
NaN-free. f64 deviation vs the old code is dominated by the `pow10` round-trip at |x|≈56:
≈ |x·ln10|·ε ≈ 1.4e-14 — clears rtol 1e-12 with two decades of margin (Tier A shipped 2.4e-14).

Call-site changes:
- `apply()` L1798-1809: drop the `apply_log10_scale` pre-scaling; pass the raw tensordot plus
  `log10_mass_scale` straight in: `log_nion = _integrate_nion_log10(td, wave[:n], log10_scale=log10_mass_scale)`;
  publish `nion = pow10(log_nion)` (transition value) **and** the new key. (`_sed_ion` feeds
  nothing else — verified.) The full-grid fallback branch uses `_integrate_nion_log10(sed_intrinsic, wave)`.
- `compute_nion` L2264-2317: new `compute_log_nion` does the slice assembly and returns the log
  form; `compute_nion` = `pow10(compute_log_nion(...))`. Existing 1e-6 parity tests keep passing.
- New `DerivedKey("log_nion", "dex", "log10(ionizing photon rate / (photons/s)); lambda < 911.76 A")`
  in `outputs()` (units-"dex" precedent: `log_mstar`).
- Gradient identity available as a test: `d log_nion / d sfh_*_log_total_mass = 1` exactly
  (weights exclude total mass — verified in `compute_nion`).

### 2. Cue high-level path → logsumexp / max-offset (cue.py `_derive_cue_params_from_ssp`)

- `total_logqion`: replace `10**logqion → weighted sum → log10` with base-10 logsumexp:
  `logsumexp(logqion_all·LN10, b=w_masked)/LN10` (`jax.scipy.special.logsumexp`; `b=0` entries drop
  out exactly, replacing the #1001 zeroing — mask non-finite table rows in the **log** domain).
  All-masked → -inf → map to the existing `-99.0` sentinel.
- `alpha_eff` / `logLratio_eff`: per-segment max-offset `m_k = max_a(log10 w_a + log_seglum_ak)`;
  `seg_w_scaled = 10**(log10 w + log_seglum − m_k)` is O(≤1); `alpha_eff` is a ratio so `m_k`
  cancels **exactly**; `log10(seg_tot_k) = m_k + log10(Σ seg_w_scaled)` → `logLratio_eff = diff(...)`.
  The `1e-300` floor (f32 no-op) is replaced by the where-dummy pattern on the scaled values.
- `i7` degenerate-case branch conditions on the log-domain sentinel instead of linear `total_qh > 0`.
- f64 parity: logsumexp vs sum-then-log is algebraically identical; numerically ~1e-15 — within bar.

### 3. Consumer migrations (complete list, each with exact shape)

| Site | Before | After |
|---|---|---|
| nebular/component.py:546-548 | `log10(maximum(nion, 1.0))` | `jnp.maximum(log_nion, 0.0)` — **exactly** equivalent incl. zero flux (`max(-inf,0)=0`) |
| nebular/component.py:735-739 + `reconstruct_nebular_phot` | `nion * 10**log_lpq` | signature takes `log_nion`; `pow10(log_nion + log_lpq)` (`-inf` propagates to 0 safely; no `+inf` operand exists). Drop the dead `jnp.sum(nion) if jnp.ndim(nion)` defensiveness (producer is scalar — verified; voice rules) |
| line_precompute.py `_nion_of_state` / L222 | `nion * lpq` | `_log_nion_of_state`; runtime `apply_log10_scale(lpq, log_nion)` — reuses the Tier A primitive, handles zero entries of `lpq` without per-entry logs. Build path (L170, eager f64) reads log and `pow10`s locally |
| radio/component.py:566,582 | `compute_l_radio_thermal(nion)` | new util `compute_l_radio_thermal_from_log_qh(log_q_h)` → `pow10(log_q_h + log10(5.5e-28))`, returns **linear** erg/s/Hz (~5e28, f32-safe; needed linearly at L583). Keep the linear public util (Condon 1992 citation) + a parity test binding the two |
| stellar `_xi_ion_fn` + `component_factory.state_to_ionizing_quantities` | `q_h / maximum(fuv·ν, _TINY)` (ν·L ≈ 1e43 → f32 inf) | `pow10(log_nion − log10(fuv_safe) − log10(nu_uv))` with where-dummy on `fuv > 0`; **identical code in both places** so `test_q_h`-style bit-equality between surfaces is preserved |
| stellar `_q_h_fn` | reads `nion` | unchanged (linear transition surface); **add** `_log_q_h_fn` reading `log_nion` |

New public property `log_q_h` (units "dex", group ionizing, doc: "log10(Q_H / (photons/s)) —
float32-safe form of q_h") registered beside `q_h` in `_SED_PROPERTIES`. Non-breaking; naming
follows `log_mstar` precedent (no existing `log_*` property, checked — this is the first, which
NAMING_CONTRACT's internal-name conventions support).

`nion` stays published during the transition: `state_to_ionizing_quantities`, `_q_h_fn`, and two
contract tests read it, and it is `pow10(log_nion)` so producer/consumers can't drift. A **two-way
inventory guard** (pattern: `tests/regression/precision/test_no_raw_flux_scale.py`) pins the exact
allowed reader set of `derived["nion"]` in `src/` so no new linear consumer can appear silently.

## Tasks (subagent-driven, tests-first)

0. **Worktree + plan doc.** New worktree off fresh `origin/main` (must contain `5769a3e46`); copy
   this plan to `docs/internal/plans/2026-07-17-float32-tier-b-log-nion.md`. All test runs use
   `PYTHONPATH=src` (never the root venv's editable install).
1. **Core integral (RED→GREEN).** New `tests/regression/precision/test_ionizing_log_domain.py`
   (inherits `regression_bug` marker from conftest): (a) frozen in-test copy of the pre-change
   `_integrate_nion` vs new wrapper, f64, rtol 1e-12, synthetic SEDs + real-SSP SEDs over
   `Z_MASS_GRID`; (b) `_integrate_nion_log10` pure-f32 finiteness + f64 parity (atol 5e-3 dex)
   under `with jax.enable_x64(False):`; (c) zero-flux → log `-inf`, wrapper exactly 0.0, gradient
   NaN-free; (d) gradient identity `d log_nion/d log_total_mass == 1` (f64, rtol 1e-9). Then
   implement §1 (core + wrapper + `apply()` rewire + DerivedKey + `compute_log_nion`).
2. **Cue derive-from-ssp (RED→GREEN).** Tests: f64 old-vs-new parity of the full returned dict
   (`gas_logqion`, `i7`) at rtol 1e-12 across ages/metallicities; pure-f32 finiteness of
   `gas_logqion` and `i7` (this is the test that today would expose the silent zeroing). Then
   implement §2.
3. **Linear consumers (RED→GREEN).** Per-site parity (f64 rtol 1e-12 old-vs-new) + pure-f32
   finiteness tests for: grid-precomp reconstruct, line-precomp runtime, radio thermal
   (`compute_l_radio_thermal_from_log_qh` vs linear util), `xi_ion`/`state_to_ionizing_quantities`.
   Then migrate all sites in §3's table.
4. **`log_q_h` property + guard.** Property registration test
   (present in `available_properties`, sugar `pred.log_q_h`, equals `log10(q_h)` where finite);
   the two-way `derived["nion"]` reader-inventory guard. **Mutation-check** the guard (insert a
   fake reader, watch it fail, remove) and at least tests 1a/2 (revert the fix in-memory, confirm
   RED) before claiming green.
5. **Restructure `test_ionizing_scale.py` test C.** Split into:
   `test_log_q_h_pure_float32_cue_only` — asserts `log_q_h` finite, f64/f32 parity ≤ 5e-3 dex,
   `gas_logqion` seam finite (PASSES — the step's deliverable); and a strict xfail
   `test_linear_observables_pure_float32_cue_only` for `q_h`/`halpha`/`balmer_decrement`
   finiteness with the reason updated to point at #1206 item 3 (`line_lums` [erg/s] storage +
   linear property units). Test A (mixed precision) must keep passing unchanged.
6. **Docs + hygiene.** Update `docs/dev/float32-tier-b-boundary.md` §1 (delivered; what remains and
   why); docstrings per tier rules (units in brackets, `.. math::` for the integral, JIT notes);
   bump `tools/file_size_allowlist.json` for touched files only (actual new counts);
   ruff check+format, `tools/check_test_markers.py`, `tools/check_british_spelling.py`, SPDX headers.
7. **Verification + ship.** See below; then commit (conventional commits), push `-u`, open PR
   referencing #1206 item 1 (labels: `area:nebular`, `area:stellar`, `area:radio`, `area:perf`,
   `enhancement`). Merge only on explicit instruction (Tier A precedent). After merge (if
   instructed): tick item 1 on #1206 with a comment.

## Verification

- Targeted: `tests/regression/precision/`, `tests/contract/test_property_catalog*.py`,
  `tests/contract/test_fast_nebular_wiring.py`, `tests/components/nebular/`,
  `tests/components/spectroscopy/test_wne_window_lut_parity.py`, `tests/components/radio/` —
  all with `PYTHONPATH=src .venv/bin/pytest ... -q`.
- Full fast tier: `.venv/bin/pytest tests/ -q` (~7.3k tests) must be 0 failures.
- **Cross-version no-behavioral-change probe** (Tier A precedent, scripts in `$CLAUDE_JOB_DIR/tmp`):
  same params through pre-change main and the branch; bar — every field bit-exact except the
  nion-chain (`q_h`, `xi_ion`, `l_thermal`, `l_nonthermal`, nebular lines/photometry through
  `gas_logqion`), which must sit ≤ 1e-12 relative (expected ~1e-14).
- Existing bit-equality contract `test_q_h` (props vs `state_to_ionizing_quantities`) must still
  pass at atol=0 — both surfaces read the same published `nion`.
- Mutation checks per Task 4.

## Risks / notes

- `exponent`-clip behavior in Cue (`exponent_safe`, L768-777) is untouched — only its
  `gas_logqion` input changes provenance.
- If Task 2's f64 parity can't hit 1e-12 on `logLratio_eff` in some corner (e.g. one segment sum
  fully degenerate), report the measured bound with the corner case rather than loosening silently.
- `pow10(log_nion)` in the eager line-precomp **build** path assumes f64 build; if a user builds
  tables under `enable_x64(False)` it would inf — add a one-line comment noting the f64-build
  assumption (full pure-f32 table build is #1206 acceptance, not this step).
