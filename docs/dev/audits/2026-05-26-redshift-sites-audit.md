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

## Plan for sub-PRs (order)

| PR | Scope | Risk |
|---|---|---|
| (this one) | Audit doc + introduce `observation/redshift_kernel.py::shift_to_obs_frame(wave_rest, L_nu_rest, z, cosmo)` as the single canonical kernel. Add parity tests vs DSPS's `precompute_ssp_obsmag_table` on the stellar slice they share. **No call-site migrations yet.** | Low — purely additive. |
| #398.b | Migrate stellar SPS precompute flux-scale duplicates to `lnu_to_fnu`. | Low — formula-equivalent. |
| #398.c | Route `predict_obs_sed` / `predict_obs_spectrum` through the unified kernel. Delete in-house duplicates in `observation/photometry.py`, `observation/spectrum.py`, `utils/grid_interp.py`, `analysis/simulate.py`. | Medium — touches the forward path. Parity tests guard. |
| #398.d | Route `WavePrecomp` build through the kernel. Verify bit-exact agreement vs the exact path. | Medium. |
| #398.e | Address inverse-cosmology pattern in AGN + nebular. Redesign precompute → predict_state interface to pass rest-frame L_ν directly. | High — architectural. May spawn its own ADR. |
| #398.f | Investigate DLA obs→rest direction. | Low — single file. |

## Count

- ~13 unique `(1+z)` multiplication/division sites in the forward path
- ~6 precompute modules with their own redshift grids
- 4 flux-scale hardcodes (2 in `stellar/sps/precompute.py`, 2 in `analysis/simulate.py`)
- 2 inverse-cosmology undoing sites (AGN + nebular components)

Total ≈ **23 call sites** the unified kernel will eventually subsume.
