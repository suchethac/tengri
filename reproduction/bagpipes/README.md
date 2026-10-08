# Reproducing BAGPIPES's physics with tengri

This folder places BAGPIPES (Carnall et al. 2018, MNRAS 480, 4379) next to tengri
block by block on shared inputs: the same BC03+MILES Kroupa SSP grid, one stellar
metallicity, one fiducial delayed-τ history with Calzetti dust and Draine & Li (2007)
infrared emission. The notebook prints every number it quotes; the Summary table is
assembled from those values and classes each remaining difference as numerical,
convention, reference code or open.

## Files

- **`01_bagpipes.py`** — the notebook, jupytext percent format.
- **`validate_matched_physics.py`** — the same matched inputs checked band by band,
  with the emission lines compared from each code's line table.
- **`_drivers/`** — code-side glue:
  - `units.py` — BAGPIPES (erg/s/Å) to tengri (erg/s/Hz), BAGPIPES's solar luminosity,
    and a bolometric round-trip check run at Setup.
  - `bagpipes_driver.py` — wrappers around `bagpipes.model_galaxy(...)`. Every model is
    built with `spec_wavs` spanning 1000-30000 Å at `R_spec = 1000` and
    `R_other = 100` elsewhere (`bagpipes.config` is changed only inside
    `_converged_sampling`), and the driver raises if the grid's median λ/Δλ over
    that range is below 1000. It also reads BAGPIPES's line table, its absorbed
    luminosity, its Inoue et al. (2014) generator and its own photometry output.
  - `bagpipes_ssp_to_dsps.py` — repackages BAGPIPES's `bc03_miles_stellar_grids.fits`
    into the DSPS HDF5 layout `load_ssp_data` reads, with absolute log10 Z labels and a
    `lsun_erg_per_s` attribute equal to BAGPIPES's solar luminosity.
  - `data/bc03_miles_from_bagpipes.h5` — the shared SSP file (not tracked).
- **`_figs/`** — generated figures, `bagpipes_NN_<what>.png` in notebook order.

## Prerequisites

```bash
pip install bagpipes jupytext jupyter
```

Only `bagpipes.model_galaxy` is used (forward modeling); the optional `pymultinest`
dependency is not needed. tengri must be importable.

### BAGPIPES 1.3.5 with NumPy 2

- `star_formation_history.py` line 51 raises `TypeError: only 0-dimensional arrays
  can be converted to Python scalars`. Change
  `self.hubble_time = utils.age_at_z[utils.z_array == 0.]` to
  `self.hubble_time = float(np.squeeze(utils.age_at_z[utils.z_array == 0.]))`.
- `np.trapz` was removed; the driver aliases it to `np.trapezoid` at import.

## Regenerating the SSP grid

```bash
python -m reproduction.bagpipes._drivers.bagpipes_ssp_to_dsps
```

| key | shape | meaning |
|---|---|---|
| `ssp_lg_age_gyr` | `(n_age,)` | `log10(age / Gyr)` |
| `ssp_lgmet` | `(n_met,)` | `log10(Z)` (absolute, not solar) |
| `ssp_wave` | `(n_wave,)` | rest-frame wavelength [Å] |
| `ssp_flux` | `(n_met, n_age, n_wave)` | L_ν per unit stellar mass, in BAGPIPES's L☉ |
| `ssp_mass_remaining` | `(n_met, n_age)` | surviving-mass fraction |

## Running

```bash
python scripts/render_reproduction_notebook.py bagpipes
```

renders the notebook, stamps it and copies it and its figures to `docs/reproduction/`.
The run takes about fifteen minutes on a CPU.

## What the notebook covers

Setup (shared SSP grid, single metallicity, BAGPIPES sampling, translation table) ·
§1 SSP templates · §2 Star formation histories · §3 Non-parametric continuity ·
§4 Composite stellar SED · §5 Metallicity · §6 Attenuation curves ·
§7 Attenuated SED and the `eta` mapping · §8 Dust emission and energy balance ·
§9 Nebular emission · §10 Line widths · §11 IGM transmission · §12 Photometry ·
§13 Panchromatic head-to-head · Summary · References.

BAGPIPES has no AGN, X-ray or radio components; see `reproduction/cigale/` for those.

## What the comparison found

The notebook's Summary table is the reference for the numbers; the rows below state what
each section shows, with the values printed by its cells.

| § | Block | Result |
|---|---|---|
| Setup | BAGPIPES sampling | A band average of the fiducial model changes by the printed amount between the build sampling and twice it. |
| §1 | SSP templates | Both codes read the same numbers; the printed maximum deviation is the float32 round trip. |
| §2 | SFH forms | Delayed-τ, constant, double power-law and lognormal match in the band ladders once `age_gyr` is BAGPIPES's age of the universe and the lognormal width is BAGPIPES's solved value. The worst bands (GALEX NUV for τ = 0.3 Gyr, GALEX FUV for the lognormal) are the BAGPIPES offsets from the reference convolution measured in §4 (BAGPIPES's age-grid integration). |
| §3 | Continuity SFH | Mass formed per bin is compared with the exact value from the ratios, edges and total mass; tengri needs BAGPIPES's bin edges through `bin_edges_gyr` (its default ladder is scaled to the source redshift). The edge-bin deviations come from tengri's default 256-point history grid and fall with `n_grid=4096`; the stellar band ratios do not depend on it (numerical). |
| §4 | Composite stellar SED | The optical median is the printed value; against a fine-grid convolution of the same SSP file the worst band of tengri deviates by at most 0.0061 and that of BAGPIPES by 0.0037 (delayed τ = 1 Gyr), 0.0176 (τ = 0.3 Gyr) and 0.0226 (lognormal); BAGPIPES integrates the history on its 0.1-dex age grid, which contributes, and the page does not isolate it as the whole cause. tengri's default metallicity scatter is an input BAGPIPES lacks and is switched off. |
| §5 | Metallicity | The two responses agree at the BC03 nodes; between nodes the two interpolation schemes differ by the printed amounts. |
| §6 | Attenuation curves | BAGPIPES's Calzetti curve uses 2.695 where Calzetti et al. (2000) give 2.659, the tengri/BAGPIPES ratio of A(λ)/A_V at A_V = 1 is 0.9924 at 1500 Å and 0.9949 at 3000 Å, and BAGPIPES's curve is 0.99947 A_V at 5500 Å. |
| §7 | Attenuated SED | Single-screen mapping holds; the largest deviations are in the GALEX bands at high A_V and follow the Calzetti coefficient. BAGPIPES's birth-cloud step and tengri's exact-mass hard step at 10 Myr are compared for young populations at `eta` ≠ 1 (printed). |
| §8 | Dust emission | tengri conserves energy; the IR-band ratios have their worst band in WISE W3 in every case (open). tengri excludes λ < 912 Å from L_abs by default, BAGPIPES does not (printed table). |
| §9 | Nebular emission | Line-table ratios for the strongest lines across log U, Z and f_esc; the low-ionization forbidden lines ([O II], [N II], [S II]) sit below BAGPIPES while the recombination lines and [O III] agree, and [N II] 6584 at Z = 0.3 Z☉ stays off 1 with BAGPIPES's [N/O] set in tengri (open); the f_esc rows follow two scalings of the lines; Hα per case-B ionizing photon is 1.021 (BAGPIPES) and 1.000 (tengri) at 1 Z☉, 0.542 and 1.070 at 2 Z☉, 0.257 and 1.047 at 2.5 Z☉ (open); the Z = 2 Z☉ line ratios follow that drop. |
| §10 | Line widths | tengri's nebular line width is set to 0 to match BAGPIPES's one-pixel lines; the fitted Hα FWHMs, the pixel sizes and the default-width case with its quadrature expectation are printed (numerical). |
| §11 | IGM | tengri equals BAGPIPES's Inoue14 generator to the printed maximum (Lyman-limit pixel); the departures from BAGPIPES's table are its 1 Å tabulation at Lyβ and its forced Lyα pixel. |
| §12 | Photometry | SED-level SDSS magnitudes and the budget; each code's own photometry at z = 0, and at z = 0.5 after the printed distance-modulus difference of the two cosmologies. |
| §13 | Head-to-head | Optical normalization with its 16-84 % spread. |
