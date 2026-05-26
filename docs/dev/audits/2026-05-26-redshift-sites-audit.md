# Audit — rest↔observer-frame transform sites in tengri

Date: 2026-05-26
Scope: `src/tengri/`
Driver: issue [#398](https://github.com/suchethac/tengri/issues/398) — delegate observer-frame SED transform to DSPS as the single source of truth.

This document inventories every site in `src/tengri/` that touches a rest↔observer-frame transform (either of wavelengths or of flux/luminosity densities), so the unified DSPS-backed kernel (introduced in subsequent PRs) can subsume them without missing any.

## Inventory

### Forward-pass redshift transforms

| File:line | What is transformed | Direction | Convention | Notes |
|---|---|---|---|---|
| `forward/filter_preintegrate.py:124` | wave grid | rest→obs | `wave_obs = ssp_wave * (1.0 + z)` | Filter preintegration; flux scale applied downstream. |
| `utils/grid_interp.py:239` | wave grid | rest→obs | `wave_obs = wave_rest * (1.0 + redshift)` | Generic template preintegration. |
| `utils/grid_interp.py:252` | effective wave | obs→rest | `eff_waves_rest = eff_waves_obs / (1.0 + redshift)` | Derives rest-frame effective wavelengths. |
| `utils/grid_interp.py:348` | line λ centres | rest→obs | `line_wavelengths_obs = line_wavelengths * (1.0 + redshift)` | Emission-line preintegration. |
| `analysis/simulate.py:277` | wave grid | rest→obs | `wave_obs = wave * (1.0 + redshift)` | Forward-model SED. |
| `analysis/simulate.py:355` | wave grid | rest→obs | `wave_obs_full = wave * (1.0 + redshift)` | Forward-model spectrum path. |
| `observation/photometry.py:142` | wave grid | rest→obs | `wave_obs = wave_rest * (1.0 + redshift)` | Filter convolution; redshifts rest SED for filter interp. |
| `observation/photometry.py:276` | wave grid | rest→obs | (same formula) | Padded filter variant. |
| `observation/spectrum.py:447` | wave grid | obs→rest | `wave_rest_query = wave_obs / (1.0 + redshift)` | Spectrum forward model; obs pixels → rest for SED interp. |

### Canonical flux-scale converters (already centralised — these are the target API for migrations)

| File:line | What | Convention |
|---|---|---|
| `utils/conversions.py:438` | `lnu_to_fnu(L_ν, z, d_L)` | `f_ν = L_ν × (1+z) / (4π d_L²)` |
| `utils/conversions.py:472` | `fnu_to_lnu(F_ν, z, d_L)` | `L_ν = F_ν × 4π d_L² / (1+z)` |

### Flux-scale duplicates (inline formula instead of calling `lnu_to_fnu`)

| File:line | Convention | Action |
|---|---|---|
| `components/stellar/sps/precompute.py:312` | `flux_scale = (1.0 + redshift) / (4π d_L²)` | Replace with `lnu_to_fnu` call. |
| `components/stellar/sps/precompute.py:505` (region) | same | Replace. |
| `analysis/simulate.py:282, 359` | `luminosity_distance(redshift)` then manual division | Could use `lnu_to_fnu` directly. |

### Inverse-cosmology "undoing" pattern (architectural mismatch flagged)

| File:line | What it does |
|---|---|
| `components/agn/component.py:331` | `inv_cosmology = 4π × 1.0² / (1.0 + z)` — backs out the flux scaling that `compute_flux_density` already applied, to recover rest-frame L_ν. |
| `components/nebular/component.py:685` | Same pattern. |

These two sites are doing an apply-then-undo dance with `compute_flux_density`. Suggests the precompute → forward-pass interface is passing the wrong quantity (observed-frame F_ν when the consumer wants rest-frame L_ν) and the components are correcting for it at the cost of clarity and one extra (1+z) round-trip per call. **Out of scope for the first PR** — file a follow-up to redesign the precompute output interface.

### Component-specific frame conventions

| File:line | Convention | Notes |
|---|---|---|
| `components/igm/component.py:160` | `wave_obs = state.wave * (1.0 + z)` for transmission lookup | Correct: Inoue+2014 grid is in observer frame. |
| `components/igm/igm.py:187, 216, 239, 296, 380, 590, 837, 851, 864` | Multiple `× (1+z)` for boundary checks | All consistent with the observer-frame convention. |
| `components/igm/dla.py:268` | `wave_rest = wave_obs / (1.0 + z_dla)` | DLA converts obs→rest for absorption — diverges from main IGM path. Check whether this is intentional or a latent bug. |
| `components/dust/emission.py:314, 360` | `T_cmb_z = T_CMB_0 × (1+z)` | This is the *physical* CMB temperature evolution, not a redshift transform of the SED. Distinct semantically — leave alone. |
| `components/radio/radio.py:227, 313` | `q × (1+z)^z_slope` | Empirical radio-FIR correlation evolution. Not a redshift transform of an SED. Leave alone. |

### Precompute modules (each has its own redshift grid + flux scaling)

| Module | Purpose | Status |
|---|---|---|
| `components/stellar/sps/precompute.py:292, 312, 467, 470, 492, 505` | Stellar photometry precompute (n_z, n_met, n_age, n_filter). | Has flux-scale duplicates. |
| `components/agn/blr_precompute.py:127, 135` | BLR line precompute. | Redshift handling consistent with main path. |
| `components/agn/kd_precompute.py:168, 256, 359, 416` | AGN spectral templates precompute. | Uses obs→rest direction (`fw_np / (1+z)`). |
| `components/nebular/{cb19,mappings_photo,mappings_shock,feltre}_precompute.py` | Nebular line precompute. | All use `× (1+z)` consistently. |

### Diagnostics + plotting (informational, not in forward path)

| File:line | What | Convention |
|---|---|---|
| `analysis/diagnostics/lines.py:93` | wave grid | `z_waves = line_rest_waves * (1.0 + redshift)` |
| `analysis/diagnostics/lines.py:160` | equivalent width | `ew_rest = ew_obs / (1.0 + redshift)` |
| `analysis/plotting/sed.py:248` | wave grid | `lam_obs = lam_rest * (1 + z)` |
| `utils/optimizations.py:194` | wave grid | `wave_obs = ssp_wave * (1.0 + redshift)` |

## Divergences from the canonical DSPS pattern

1. **No DSPS calls in the forward-pass redshift transforms.** Every site is hand-rolled. (This is the entire reason for #398.)
2. **Flux-scale duplicates.** `lnu_to_fnu` exists and is canonical; two precompute sites inline-duplicate the formula. **Quick win** — replace with the function call.
3. **Inverse-cosmology pattern in AGN + nebular.** Suggests the precompute output interface is misaligned with what `predict_state` consumers want. Out of scope for the unified-kernel PR; file follow-up.
4. **DLA direction inverted vs main IGM.** Worth a closer look — may be a latent bug.
5. **No effective-wavelength contract.** `grid_interp.py:248–252` recomputes `eff_waves_rest = eff_waves_obs / (1+z)` per call.

## Post-merge reassessment (2026-05-26, after #402 + #403 landed)

After landing the unified kernel (#402) and the one real bug it surfaced (#403), several of the audit's original recommendations turned out **not to apply as written**:

| Original recommendation | Reality | Status |
|---|---|---|
| Migrate `stellar/sps/precompute.py:312, 505` to `lnu_to_fnu` | Doesn't fit. Both sites need the **scalar** `(1+z)/(4π d_L²)` factor stored on a precompute dataclass, not the multiplied `L × factor` result. The inline comment at lines 304–310 documents why inlining was chosen. | **Skip** |
| Route `predict_obs_sed` through the kernel | Doesn't apply. `predict_obs_sed` returns L_ν on observed-frame wavelengths — relabels the wavelength axis but does **not** perform the L→F conversion. The unified kernel produces F_ν, a different output. | **Skip — already correctly factored** |
| Migrate `compute_flux_density` in `observation/photometry.py` | Already uses canonical `lnu_to_fnu` at line 152. Not a duplicate. | **Skip** |
| DLA obs→rest direction is a possible bug | Not a bug. The Voigt cross-section is naturally in the absorber rest frame; `dla_transmission_obs` correctly de-redshifts. IGM (Inoue+2014) is tabulated in observed-frame coordinates — different physics, different conventions, both correct. Docstring already documents this. | **Skip — close as "expected divergence"** |
| `analysis/simulate.py:282, 359` — fix `dl_cm=1.0` fallback | **Real bug.** The Python-side `if redshift > 0 else 1.0` guard was a ~10^19× flux error at z=0; `luminosity_distance(0)` already returns 10 pc via the absolute-magnitude convention. | **Fixed in #403** |
| AGN + nebular inverse-cosmology pattern | Real architectural issue (apply-then-undo dance with `compute_flux_density`). Needs an ADR before code. | **Open — separate PR** |

### What the unified kernel actually buys us

`shift_to_obs_frame` from #402 is the **reference implementation**, not a wholesale replacement of existing code. Most existing call sites are already canonical or do trivial `wave * (1+z)` relabeling. Its real value:

1. **One citable convention.** Future code that needs a rest→obs SED transform has one canonical entry point.
2. **DSPS parity lock.** The contract tests at `tests/contract/test_redshift_kernel_dsps_parity.py` catch any future drift from DSPS's convention.
3. **Foundation for the AGN + nebular architectural fix.** When the inverse-cosmology pattern is cleaned up, the new code routes through this kernel rather than re-implementing it.

### Honest accounting of the "23 sites"

The original audit's "23 call sites" overstated the available migration scope. The true breakdown:

- **~3 sites** genuinely benefit from the kernel — the AGN + nebular components, once the inverse-cosmology pattern is addressed (sub-PR #398.e, blocked on ADR).
- **~5 sites** are already canonical (use `lnu_to_fnu`, `luminosity_distance`). No migration needed.
- **~10 sites** are inline `wave * (1+z)` wavelength relabeling. Too small to merit a function call.
- **~5 sites** have valid reasons to inline (scalar storage with documented constraints, numpy build-time loops outside JIT).
- **1 site** had a real bug — fixed in #403.

Of the original 6 sub-PRs proposed below, only **#398.e** (AGN + nebular architectural cleanup) is worth pursuing as scoped. The rest were resolved either by being already-correct, by the single bug fix in #403, or by being closed as expected divergences.

## Plan for sub-PRs (original, before reassessment — kept for posterity)

| PR | Scope | Risk | Outcome |
|---|---|---|---|
| #402 | Audit doc + introduce `observation/redshift_kernel.py::shift_to_obs_frame(wave_rest, L_nu_rest, z, cosmo)` as the single canonical kernel. Add parity tests vs DSPS. | Low — purely additive. | ✅ Merged |
| #398.b | Migrate stellar SPS precompute flux-scale duplicates to `lnu_to_fnu`. | Low — formula-equivalent. | ❌ Skipped (sites need scalar storage; see reassessment) |
| #398.c | Route `predict_obs_sed` / `predict_obs_spectrum` through the unified kernel. Delete in-house duplicates in `observation/photometry.py`, `observation/spectrum.py`, `utils/grid_interp.py`, `analysis/simulate.py`. | Medium. | ❌ Mostly skipped (`predict_obs_sed` returns L_ν not F_ν; `compute_flux_density` already canonical; `simulate.py` z=0 bug fixed in #403) |
| #398.d | Route `WavePrecomp` build through the kernel. | Medium. | ⏸ Deferred (kernel is the citable convention; precompute path already correct) |
| #398.e | Address inverse-cosmology pattern in AGN + nebular. | High — architectural. ADR. | 🔜 **Open** — the one remaining real win |
| #398.f | Investigate DLA obs→rest direction. | Low — single file. | ❌ Skipped (not a bug — physics-driven convention) |

## Count

- ~13 unique `(1+z)` multiplication/division sites in the forward path
- ~6 precompute modules with their own redshift grids
- 4 flux-scale hardcodes (2 in `stellar/sps/precompute.py`, 2 in `analysis/simulate.py`)
- 2 inverse-cosmology undoing sites (AGN + nebular components)

Total ≈ **23 call sites** the unified kernel will eventually subsume.
