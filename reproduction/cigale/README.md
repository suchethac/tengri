# Reproducing CIGALE's physics with tengri

This folder reproduces the physics modules of CIGALE
(Boquien et al. 2019, A&A 622, A103) using tengri, component by
component.

## Files

- **`01_cigale.py`** — the notebook (jupytext percent format).
- **`_drivers/`** — code-side glue:
  - `units.py` — CIGALE (W/nm on nm) ↔ tengri (erg/s/Hz on Å).
  - `cigale_driver.py` — instantiate `pcigale.sed_modules` and read
    out SEDs, attenuation curves, SFH curves.
  - `cigale_ssp_to_dsps.py` — one-off repackaging of CIGALE's BC03 Chabrier
    templates into the DSPS HDF5 layout tengri reads.
  - `consistency_audit.py` — wavelength-resolved CIGALE vs tengri
    ratio statistics for every section, runnable on its own.
- **`_drivers/data/bc03_from_cigale.h5`** — the shared SSP file. Both
  codes consume this; §1 residuals below floating-point precision are
  interpolation, nothing else.
- **`_figs/`** — generated figures.

## Prerequisites

```bash
# pcigale is not on PyPI under any name; it builds its template database at
# install time, which is why the clone is ~2.8 GB and the build is not quick.
git clone https://gitlab.lam.fr/cigale/cigale.git && cd cigale
# CIGALE 2025.1 predates numpy 2, which removed np.trapz in favour of the
# identically-signed np.trapezoid. 58 call sites, no other numpy-2 breakage:
grep -rl 'np\.trapz' --include='*.py' . | xargs sed -i '' 's/np\.trapz\b/np.trapezoid/g'
pip install --no-build-isolation .
cd .. && pip install jupytext jupyter

python -m reproduction.cigale._drivers.cigale_ssp_to_dsps   # writes the shared SSP grid
```

The last line is not optional. `_drivers/data/bc03_from_cigale.h5` is 415 MB and
excluded by `.gitignore`'s `*.h5`, so a fresh checkout does not have it and the
notebook stops at its second cell with `SystemExit`. Under `nbclient` that reads
as a *clean* stop, not an error: an automated run reports zero failures, writes
a notebook with 5 of its 18 figures and none of its §-lines, and exits 0. Check
that the run took minutes rather than seconds.

Tengri itself should already be available on `PYTHONPATH`.

## Running

```bash
jupytext --to ipynb 01_cigale.py
jupyter nbconvert --to html --execute 01_cigale.ipynb
```

Expected runtime: 10–15 minutes on a CPU. First-time JAX compilation
for the Cue nebular emulator and the AGN blocks dominates; subsequent
runs use the persistent cache and finish in well under a minute.

## Regenerating the BC03 templates

```bash
python _drivers/cigale_ssp_to_dsps.py \
  --input /path/to/.venv/lib/python3.12/site-packages/pcigale/data/bc03/ \
  --output _drivers/data/bc03_from_cigale.h5
```

The output HDF5 has the DSPS-compatible shape:

| key | shape | meaning |
|---|---|---|
| `ssp_lg_age_gyr` | `(n_age,)` | `log10(age / Gyr)` |
| `ssp_lgmet` | `(n_met,)` | `log10(Z)` (absolute, not solar) |
| `ssp_wave` | `(n_wave,)` | rest-frame wavelength [Å] |
| `ssp_flux` | `(n_age, n_met, n_wave)` | L_λ at unit stellar mass |

Both `pcigale.sed_modules.bc03` and `tengri.load_ssp()` consume the
same file with no per-side modifications.

## What the notebook covers

§1 SSP · §2 SFH · §3 stellar SED · §4 dust attenuation curves ·
§5 attenuated stellar SED · §6 dust IR + energy balance ·
§7 panchromatic · §8 nebular · §9 AGN · §10 X-ray · §11 radio ·
§12 IGM.
