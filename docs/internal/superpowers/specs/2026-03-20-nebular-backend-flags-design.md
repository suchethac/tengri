# Nebular Backend Selection Redesign

**Date:** 2026-03-20
**Status:** Approved
**Scope:** ParamSpec nebular flags, Model backend dispatch, Cue ionization sources

## Problem

The current interface uses a single `nebular` flag (bool/str) plus `cloudy_grid_path` / `cue_weights_path` kwargs. This makes backend selection implicit and error-prone:
- Users must know the right combination of flags and paths
- No validation that the SSP isochrone matches the CLOUDY grid
- Cue Q_H mass scaling and metallicity conversion were manual (now fixed, but the interface should prevent future mistakes)
- No way to express ionization source (SSP vs AGN vs both)

## Design

### Backend Selection: Three Mutually Exclusive Flags

| Flag | Backend | Free params | Requires |
|------|---------|-------------|----------|
| `nebular_ssp=True` | BakedIn | None | wNE SSP file |
| `nebular=True` | CloudyGrid | logU, logZ_gas, fesc | `cloudy_grid_path` |
| `nebular_cue=True` | Cue | logU, logZ_gas, fesc, (ionspec_*) | None (default weights) |

**Conflict rules:**
- Multiple flags set → `ValueError` listing the conflict
- `nebular_ssp=True` + user sets `neb_logU` etc. → `UserWarning`: param ignored, emission is baked into SSP at fixed logU/logZ
- `nebular=True` without `cloudy_grid_path` → `ValueError` listing available grids in `data/`

**Backward compatibility:**
- `nebular="cue"` → maps to `nebular_cue=True`
- `nebular="cloudy"` → maps to `nebular=True` (still needs path)
- `cue_weights_path=...` without explicit flag → `nebular_cue=True`
- `cloudy_grid_path=...` without explicit flag → `nebular=True`

### CLOUDY Grid Path Error Message

When `nebular=True` but no `cloudy_grid_path`:

```
ValueError: nebular=True requires cloudy_grid_path. Available grids:
  cloudy_grid_mist.h5      (MIST isochrones)
  cloudy_grid_mist_wd.h5   (MIST + white dwarfs)
  cloudy_grid_prsc.h5      (PARSEC isochrones)
  cloudy_grid_prsc_wd.h5   (PARSEC + white dwarfs)
  cloudy_grid_pdva.h5      (PADOVA isochrones)
  cloudy_grid_bpss.h5      (BPASS isochrones)
Match the grid isochrone to your SSP for consistency.
```

The grid listing is discovered dynamically from `data/cloudy_grid_*.h5`.

### Cue Weights Default Path

Resolved from package structure: `Path(__file__).resolve().parents[2] / "data" / "cue_weights.npz"`. User can override with `cue_weights_path=...`.

### Ionization Source Flag (Cue only)

```python
neb_ionization = "ssp"       # default: young stars only
neb_ionization = "agn"       # AGN disc only (future)
neb_ionization = "ssp+agn"   # both summed (future)
```

**Current scope (this PR):** Only `"ssp"` is implemented. Setting `"agn"` or `"ssp+agn"` raises `NotImplementedError("AGN ionization requires agn_model — see future PR")`.

For `"ssp"` mode, CueBackend uses `_compute_weighted_cue_params` to derive Q_H and ionizing spectrum from SSP weights (already implemented).

### Cue Ionizing Spectrum Parameters

Three modes based on what the user provides:

1. **Not set (default):** SSP-derived power-law approximation via `_compute_weighted_cue_params`. Info message logged: `"Cue: ionizing spectrum derived from SSP (4-segment power-law fit)"`

2. **Fixed values:** `ionspec_index1=Fixed(5.0)` etc. Used directly, bypasses SSP derivation for those params. No warning.

3. **Free params:** `ionspec_index1=Uniform(1, 42)` etc. Registered as inference parameters. The Cue forward model uses them self-consistently during sampling.

### Registered Parameters by Backend

| Parameter | nebular_ssp | nebular (CLOUDY) | nebular_cue |
|-----------|:-----------:|:----------------:|:-----------:|
| neb_logU | - | Yes | Yes |
| neb_logZ_gas | - | Yes | Yes |
| neb_fesc | - | Yes | Yes |
| neb_fesc_lya | - | Yes | Yes |
| ionspec_index1..4 | - | - | Optional |
| ionspec_logLratio1..3 | - | - | Optional |
| gas_logn | - | - | Optional |
| gas_logno | - | - | Optional |
| gas_logco | - | - | Optional |

"Optional" = only registered if the user explicitly provides a distribution (Fixed/Uniform/etc.).

## Files Changed

1. **`param_spec.py`**: New flags, validation, param registration logic
2. **`model.py`**: Backend dispatch using new flags (simplified)
3. **`cue.py`**: Default weights path constant
4. **`__init__.py` (nebular)**: Export `_DEFAULT_CUE_WEIGHTS_PATH`

## Testing

- Unit tests for all flag combinations and conflict detection
- Unit tests for missing grid path error with dynamic grid listing
- Unit tests for Cue ionspec param registration (none/fixed/free)
- Existing crossval tests must still pass unchanged
