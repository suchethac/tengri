# Reproducing BAGPIPES with tengri

This folder places BAGPIPES (Carnall et al. 2018, MNRAS 480, 4379) next
to tengri component by component. Same parameters, same units, same
SSP grid; one figure per physics block.

## Files

- **`01_bagpipes.py`** — the notebook, jupytext percent format.
- **`_drivers/`** — code-side glue:
  - `units.py` — bagpipes (erg/s/Å) ↔ tengri (erg/s/Hz). Ships
    `verify_unit_conversion(rtol=1e-3)`; the notebook trips at Setup
    if the converter ever drifts.
  - `bagpipes_driver.py` — thin wrappers around
    `bagpipes.model_galaxy(...)` to extract stellar / attenuated /
    nebular SEDs, SFH curves, and IGM transmission in tengri's units.
  - `bagpipes_ssp_to_dsps.py` — one-off port of bagpipes' bundled
    `bc03_miles_stellar_grids.fits` (BC03 templates in the MILES
    extended-wavelength library, Kroupa IMF) into the DSPS HDF5
    layout tengri's `load_ssp_data` reads. Includes the
    `LIV_MSTAR_FRAC` table so the mass-loss bookkeeping is
    bit-exact.
  - `data/bc03_miles_from_bagpipes.h5` — the shared SSP file. Both
    codes consume this; §1 residuals below floating-point precision
    are interpolation, nothing else.
- **`_figs/`** — generated figures.

## Prerequisites

```bash
pip install bagpipes jupytext jupyter
```

Bagpipes' optional `pymultinest` dependency is only needed for
posterior sampling and can be ignored here — we use
`bagpipes.model_galaxy` for forward-modelling only. Tengri itself
should already be importable.

## Regenerating the BC03+MILES SSP grid

```bash
python -m reproduction.bagpipes._drivers.bagpipes_ssp_to_dsps
```

The output HDF5 has the DSPS-compatible shape:

| key | shape | meaning |
|---|---|---|
| `ssp_lg_age_gyr` | `(n_age,)` | `log10(age / Gyr)` |
| `ssp_lgmet` | `(n_met,)` | `log10(Z)` (absolute, not solar) |
| `ssp_wave` | `(n_wave,)` | rest-frame wavelength [Å] |
| `ssp_flux` | `(n_met, n_age, n_wave)` | L_ν at unit stellar mass |
| `ssp_mass_remaining` | `(n_met, n_age)` | surviving-mass fraction |

Both `bagpipes.models.model_galaxy` and `tengri.load_ssp_data`
consume the same numeric arrays — `ssp_flux` is derived from
bagpipes' `bc03_miles_stellar_grids.fits` by a single
`L_λ × λ²/c` Jacobian.

## Running

```bash
jupytext --to ipynb 01_bagpipes.py
PYTHONPATH=$PWD/../..:$PWD/../../src \
  jupyter nbconvert --to html --execute 01_bagpipes.ipynb \
  --ExecutePreprocessor.timeout=900
```

Expected runtime: 5–10 minutes on a CPU. First-time JAX compilation
for the Cue nebular emulator dominates; subsequent runs reuse the
persistent cache and finish in under a minute.

## What the notebook covers

§1 SSPs · §2 delayed-τ SFH · §3 stellar SED · §4 dust attenuation
curves · §5 attenuation applied · §6 dust IR + energy balance ·
§7 panchromatic · §8 nebular (Cloudy v25 vs Cue v17) · §9 LSF /
velocity broadening · §10 double-power-law SFH · §11 lognormal SFH ·
§12 IGM.

AGN, X-ray, and radio sections are skipped — BAGPIPES has no
counterpart. See `reproduction/cigale/` for those.

## What the comparison found

A 1:1 comparison is the best bug-discovery tool we have. Each block
either reproduces the reference, or surfaces a gap.

| § | Block | Result |
|---|---|---|
| 1 | BC03+MILES SSP | Float32 round-trip floor (~1e-5 typical, ~1e-7 best). |
| 2 | Delayed-τ SFH | Both integrate to `10^massformed`. |
| 3 | Stellar SED | tengri / BAGPIPES = 1.010 ± 0.001 in the optical (1% systematic — under investigation). |
| 4 | Dust attenuation curves | Calzetti, Cardelli, CF00, Salim — visual match. |
| 5 | Attenuation applied | Matched at single-Av. |
| 6 | DL07 dust IR + energy balance | Exact (`L_IR_emitted − L_absorbed = 0` to floating point). |
| 8 | Nebular | tengri Cue Hα ≈ 3.6× BAGPIPES Cloudy v25 Hα at matched SFR and logU. Most of the gap is Cloudy v17 (Cue training set) vs Cloudy v25 (current BAGPIPES) plus bare-stellar vs SFH-integrated ionising-luminosity paths. |
| 9 | LSF / velocity broadening | tengri `velocity_broaden` matches the analytic Gaussian σ = 150 km/s FWHM (7.78 Å vs 7.73 Å expected) to 0.7 %. BAGPIPES gives 9.5 Å — its native R_spec = 1000 carries ~127 km/s of resolution that adds in quadrature with `veldisp`. Both behaviours are correct; they bracket different conventions of "intrinsic line width". |
| 10 | Double power-law SFH | Same closed-form shape on both sides, **but applied in different time frames**: BAGPIPES treats `t` as cosmic age since the Big Bang, tengri treats it as lookback since formation. For matched `(α, β, τ)` the two curves are time-reversed images of each other. Not a bug — a convention difference researchers reading two papers should know about. |
| 11 | Lognormal SFH | Same shape, same time-frame caveat as §10. BAGPIPES `tmax` ≡ cosmic age; tengri `peak_lbt_gyr` ≡ lookback time. |
| 12 | Inoue14 IGM | Within ~1e-3 between 950–1216 Å. Tengri returns 0 below the Lyman limit (912 Å) — bagpipes returns the smooth continuum predicted by Inoue+2014. **Filed as a tengri bug.** |

## Open follow-ups surfaced by this comparison

Seven tengri issues were filed while writing this notebook. Each is
small (≤ 50 LOC) and unblocked from this PR.

- **igm**: `inoue14` returns 0 below the Lyman limit; the Inoue+2014
  paper's smooth LyC continuum is missing.
- **parameters**: `list_dust_emission_models()` advertises aliases
  (`dl07`, `dl14`, `mbb`) that the `SEDModel.build` dust.emission
  validator rejects — same registry-drift pattern as the AGN fix in
  the CIGALE notebook.
- **public-API**: `load_ssp` / `load_ssp_data` is the only path to
  bring an SSP HDF5 into a notebook and is not in `tengri.*`. Every
  example reaches into `tengri.components.stellar.sps.dsps_wrapper`.
- **public-API**: `igm_transmission(wave_obs, z)` is the dispatcher
  for Inoue14 / Madau / Meiksin but not in `tengri.*`. The reproduction
  notebook imports from `tengri.components.igm`.
- **public-API**: `velocity_broaden(flux, wave, sigma_km_s)` is the
  fast JIT'd Gaussian LSF kernel matching the analytic σ to ~1 %.
  Not in `tengri.observation.*`. BAGPIPES users need a public path
  to apply their `veldisp` / `R_curve` to a tengri spectrum.
- **investigation**: §3 reports a flat 1.010 × tengri/BAGPIPES ratio
  in the optical even though both codes consume the **same** SSP
  numerics and form the **same** total mass. Likely a quadrature or
  surviving-mass-fraction-convention residual; root cause TBD.
- **dust**: BAGPIPES' VW07 two-component law (independent slopes for
  birth-cloud and diffuse) has no tengri counterpart. The closest is
  `two_component` with a shared slope — a gap for BAGPIPES users
  who fit with VW07.

Any percent-level disagreement that does not have a one-sentence
physics explanation in the figure caption above the audit table is
either tracked here or filed as an open issue.
