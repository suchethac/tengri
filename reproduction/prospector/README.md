# Reproducing Prospector's physics with tengri

This folder places Prospector (Johnson, Leja, Conroy & Speagle 2021,
ApJS 254, 22) next to tengri component by component. Same parameters,
same units, same SSP grid; one figure per physics block.

Prospector's forward model is FSPS (Conroy, Gunn & White 2009), called
through `python-fsps`, with dust-attenuation curves from `sedpy`. The
left panel of every figure is that engine, run live; the right panel is
tengri. Both read the *same* FSPS MIST + MILES Chabrier templates, so a
§1 residual below floating-point precision is interpolation alone.

## Files

- **`01_prospector.py`** — the notebook, jupytext percent format.
- **`_drivers/`** — code-side glue:
  - `units.py` — FSPS (L⊙/Hz) ↔ tengri (erg/s/Hz). Ships
    `verify_unit_conversion(rtol=1e-3)`; the notebook trips at Setup if
    the converter ever drifts.
  - `prospector_driver.py` — thin wrappers around `fsps.StellarPopulation`
    (the engine Prospector holds in `CSPSpecBasis.ssp`) and
    `sedpy.attenuation`, returning SSPs, composite SEDs, SFHs, dust /
    nebular / AGN / IGM curves in tengri's units.
  - `data/` — the downloaded SSP grid lands here and is **git-ignored**;
    nothing in this folder is committed.
- **`_figs/`** — generated figures.

## Prerequisites

```bash
git clone https://github.com/cconroy20/fsps.git && git -C fsps checkout a431c59
export SPS_HOME=$PWD/fsps
FFLAGS="-DMILES=1 -DC3K_LR=0 -DC3K_HR=0 -DMIST=1" \
  pip install --no-binary fsps --no-deps 'fsps==0.4.7'
codesign --force -s - "$(python -c 'import fsps,glob,os;print(glob.glob(os.path.join(os.path.dirname(fsps.__file__),"_fsps*.so"))[0])')"   # Apple Silicon only
pip install astro-prospector sedpy jupytext jupyter
```

Four things, each of which fails silently on its own. The check that they all
landed is the header the notebook prints:
`libraries ['mist', 'miles', 'DL07']`, `nwave 5994`, `nz 12`.

**The FFLAGS, in full.** The spectral library is fixed when the Fortran is
compiled, and `src/sps_vars.f90` guards with `#ifndef`, so `-DMILES` *alone*
leaves `C3K_LR` defined too and you get a binary carrying both. tengri's grid is
MIST+MILES; against a C3K build §1's SSP residual reads ~1e-1 instead of
~1e-9 and every later section moves with it, with nothing raised.
`prospector_driver` now refuses a non-MILES build outright.

**python-fsps 0.4.7, not newer.** It bundles its own copy of the FSPS Fortran
and uses `SPS_HOME` for *data only*, so the code and the data have to be of one
era. 0.4.7 was current on the audit date; 0.5.0 against audit-era data dies with
`Fortran runtime error: End of file` on `ISOCHRONES/MIST/zlegend.dat`.

**FSPS at `a431c59`.** Commit `05b5e55` ("Remove the now unused, old C3K
zlegend") landed after the audit and drops a metallicity, so HEAD reports
`nz 13` and the residual does not return to 1e-9.

**The codesign, on Apple Silicon.** A locally built `_fsps*.so` that is not
signed is SIGKILLed on import. The crash report says `namespace: CODESIGNING`
and `EXC_BAD_ACCESS (Code Signature Invalid)`; from the outside it looks exactly
like the OOM kill you would expect from a 2 GB Fortran build, so check the
report before blaming memory.

Without `SPS_HOME` the notebook stops at Setup with a message. The
notebook needs an FSPS compiled with **MIST isochrones + MILES spectral
library**; `sp.libraries` reports the combination your build uses, and a
matching build prints `['mist', 'miles', ...]` with 5994 wavelengths over 12
metallicities. (MILES was python-fsps's default once; it is not any more.)

tengri downloads the matching grid automatically (next section). If your FSPS
uses a different isochrone or library, swap the catalog name in the "Common
SSP grid" cell so the two sides use identical inputs.

## The SSP grid (downloaded, not shipped)

The notebook calls

```python
tengri.download_ssp("fsps_mist_miles_chabrier", dest="_drivers/data")
```

which fetches the ready-made DSPS-shaped HDF5 from the public
catalog (`tengri.list_known_ssps()` lists the ~20 available
isochrone × library × IMF combinations). The grid is **bare-stellar**
(no baked-in nebular emission), which is what the Cue nebular emulator
in §8 requires. The file is cached under `_drivers/data/` and ignored by
git — Prospector/FSPS templates are never committed here.

## Running

```bash
jupytext --to ipynb 01_prospector.py
PYTHONPATH=$PWD/../..:$PWD/../../src SPS_HOME=/path/to/fsps \
  jupyter nbconvert --to html --execute 01_prospector.ipynb \
  --ExecutePreprocessor.timeout=1800
```

Expected runtime: a few minutes on a CPU. Building the FSPS
`StellarPopulation` (~30 s, cached for the session) and the first-time
JAX compilation of the Cue nebular emulator dominate; subsequent runs
reuse the persistent cache.

## What the notebook covers

§1 SSPs · §2 delayed-τ SFH · §3 stellar SED · §4 dust attenuation
curves (Calzetti, Charlot & Fall, Kriek & Conroy) · §5 attenuation
applied · §6 dust IR + energy balance (Draine & Li 2007) · §7
panchromatic · §8 nebular (FSPS Byler+2017 vs tengri Cue) · §9 AGN
(Nenkova 2008 torus) · §12 IGM (Madau 1995).

X-ray and radio are skipped — Prospector has no counterpart. The
section numbering keeps the gap (§10 X-ray, §11 radio) so it lines up
with the CIGALE master sequence; see `reproduction/cigale/` for those.

## What the comparison found

The per-section scalars printed by the notebook (residuals, ratios,
peak locations) are the quantitative record; the figures in `_figs/`
are the visual one. In short: the shared SSP grid, the closed-form SFH
shape, the attenuation curves, the dust-IR energy balance, and the
Madau IGM reproduce FSPS to floating point or a fraction of a percent.
The nebular block is the deliberate exception — FSPS's Byler+2017
Cloudy grid and tengri's Cue emulator use different photoionization
inputs, so the Hα ratio is reported rather than forced to agree.

Two *default conventions* differ and are switchable (#961): tengri's
energy balance excludes the Lyman continuum from dust heating (#922),
so its far-IR sits ~11 % below FSPS at the star-forming fiducial —
`dust={'eb_include_lyc': True}` opts into the FSPS convention (§6
prints both ratios). And tengri defaults the IGM **on** (Inoue+2014)
where Prospector defaults it off — use `igm={'type': 'none'}` (or
`'madau'` with FSPS's flag set) when matching band fluxes at z ≳ 1.
