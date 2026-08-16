# Fully composable AGN model — physics-correct unified AGN via the grammar

**Date:** 2026-06-04 · **Branch:** cs/synthesizer-parity · **Status:** approved (Approach A)

## Goal

Reproduce everything the monolithic `unified_nlr_blr` model does — disc + torus +
NLR + BLR + FeII + attenuation + **Type-1/2 inclination masking** — purely through
the composable `agn={'type':'composable', ...}` grammar, with physically correct
differential masking: the disc and BLR are obscured as the sightline grazes the
torus, while the spatially-extended **NLR stays isotropic (unmasked)**. Ship a
turnkey recipe and verify the composable config matches `unified_nlr_blr`.

## Why now

The composable `lines` slot is single-valued and the runner's masking stage
(`compose_l_nu` Stage 4.5) masks the whole central engine (disc + lines + FeII)
*uniformly*. With the new combined `nlr_blr` block (PR #661) summing NLR+BLR into
one array, that uniform screen would wrongly obscure the NLR. `unified_nlr_blr`
(reference) instead applies a gray `_sigmoid_mask(cos_inc, theta_torus)` to
disc + BLR only, leaving NLR and torus unmasked.

## Design (Approach A — back-compatible lines split + per-torus gray mask)

### 1. Lines-block isotropic split (back-compatible contract extension)

A `lines` block MAY return either:
- `L` — a single `L_lambda` array (current contract; treated as fully
  **anisotropic** / maskable), or
- `(L_maskable, L_isotropic)` — a 2-tuple splitting maskable (BLR-like) from
  isotropic (NLR-like) emission.

Block-by-block:
| block | returns |
|---|---|
| `blr`, `blr_synthesizer`, `grahsp`, `qsogen`, `none` | `L` (anisotropic — unchanged) |
| `nlr`, `nlr_synthesizer` | `(0, L_nlr)` (isotropic) |
| `nlr_blr`, `nlr_blr_synthesizer` | `(L_blr, L_nlr)` |

Runner helper `_split_lines(result) -> (aniso, iso)`: tuple → unpack; array →
`(array, zeros_like)`. **Back-compat:** a block returning a bare array behaves
exactly as today.

### 2. Gray visibility mask in the runner (per-torus, no double-counting)

Reuse `_sigmoid_mask(agn_cos_inc, agn_theta_torus)` (move/import from
`unified.py` into a shared masking module). The masking stage applies to the
**anisotropic** central engine only; isotropic NLR is added *after* the screen:

```
L_aniso = L_disc + L_lines_aniso + L_feii
screen  = torus_screen_transmission(...)        if torus in TORUS_SCREEN_PARAMS  # dusty, λ-dependent (fritz/skirtor)
        = sigmoid_visibility_mask(cos_inc, θ_t) elif torus != "none"             # gray geometric (unified)
        = 1.0                                    else
L_central = L_aniso * screen + L_lines_iso        # NLR bypasses the screen
L_total   = (L_central + L_torus) * atten_factor  # host/foreground screen still hits NLR
```

**One obscuration model per torus** avoids double-counting: dusty-screen tori
(`skirtor`, `fritz`) keep their wavelength-dependent screen; every other non-`none`
torus gets the gray geometric mask.

### 3. New declared param `agn_theta_torus`

Torus half-opening angle [deg], `Uniform(0, 90, default=30.0)`. Drives the gray
mask's critical angle `inc_crit = 90° − theta_torus`. Default `theta_torus=30°`
with default `agn_cos_inc=0.866` (30°) gives vis ≈ 1 → **existing models
unchanged**.

### 4. `recipes.unified_agn()`

Turnkey composable config reproducing `unified_nlr_blr`'s component set:
disc=`multicolor`, torus=`simple` (1000 K graybody) or `silva04`, lines=`nlr_blr`
(analytic) / `nlr_blr_synthesizer` (grid), with `agn_cos_inc` + `agn_theta_torus`
free. Docstring states the SSP requirement and that it is the composable
equivalent of the monolithic `unified_nlr_blr`.

### 5. CONSUMES updates

`agn_theta_torus` (and `agn_cos_inc` where absent) added to the consumed sets of
the gray-mask torus blocks (`simple`, `silva04`, `nenkova`, `two_temperature`,
`cat3d_wind`, …) so `agn={'*': FREE}` frees the masking knobs for a unified config.

## Testing

- **Lines split (CI-safe):** each block returns the documented shape; combined
  block split sums to the single-region totals.
- **Differential masking (CI-safe, analytic + synthetic SSP):** sweep
  `agn_cos_inc` for a unified config; the disc + BLR continuum drops past
  `inc_crit` while the NLR line flux stays flat (ratio ~1).
- **Parity vs `unified_nlr_blr`:** the composable `unified_agn()` config matches
  the monolithic model's inclination behavior (sign + transition angle) to a
  documented tolerance; not bit-exact (composable energy-couples disc→torus).
- **Back-compat:** default-inclination existing AGN configs unchanged
  (predict equal pre/post within 1e-6).

## Out of scope

- Bit-exact numerical parity with `unified_nlr_blr` (different disc/torus
  coupling — documented).
- Retiring the monolithic models (kept as expert shorthand).
- Visibility-weighted polar-dust reddening of the BLR (the existing `polar_dust`
  atten block stays as-is; NLR simply bypasses the inclination screen).

## Blast radius

`lines_blocks.py` (return shapes), `runner.py` (`_split_lines` + masking stage),
`_params.py` (+`agn_theta_torus`), a new `masking` helper, `_consumes.py`,
`recipes`, tests. All lines-contract changes are back-compatible (bare array ==
anisotropic).
