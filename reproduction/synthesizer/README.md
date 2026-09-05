# Reproducing Synthesizer's physics with tengri

This folder places Synthesizer (Lovell et al. 2025, OJA, doi:10.33232/001c.145766;
Roper et al. 2026, JOSS, doi:10.21105/joss.09436 — **cite both**) next to tengri
component by component — same parameters, same
units, one figure per physics block — with a focus on the **Unified AGN** model
(§9). tengri and Synthesizer are **independent codes**; this is a peer-to-peer
comparison, exactly like the CIGALE, Prospector, and BAGPIPES notebooks.

Synthesizer builds spectra by extracting from pre-computed HDF5 grids and walking
an `EmissionModel` tree. The left panel of every figure is Synthesizer, driven
through `synthesizer.*` (`Grid`, `parametric.Stars`, `BlackHole`, `UnifiedAGN`);
the right panel is tengri. For the stellar SSP (§1) both read the *same*
templates — Synthesizer's `test_grid` `incident` spectra re-shaped into tengri's
DSPS HDF5 — so the §1 residual is interpolation alone. For the AGN line regions
(§9c/§9d) tengri reads the *same* Synthesizer Cloudy AGN grids through its
`compute_nlr_sed_synthesizer` / `compute_blr_sed_synthesizer` adapters.

## Files

- **`01_synthesizer.py`** — the notebook, jupytext percent format.
- **`_drivers/`** — code-side glue:
  - `units.py` — Synthesizer (`unyt` erg/s/Hz) → tengri (plain erg/s/Hz). Ships
    `verify_unit_conversion(rtol=1e-3)`; the notebook trips at Setup if the
    converter drifts.
  - `synthesizer_driver.py` — thin wrappers around the Synthesizer forward model
    (stellar grids, parametric SFH, attenuation/dust/IGM laws, and the
    `UnifiedAGN` black-hole emission tree), returning everything in tengri's
    convention. Heavy grids are cached at module level.
  - `synthesizer_ssp_to_dsps.py` — one-off repackaging of Synthesizer's stellar
    `incident` grid into a DSPS-shaped HDF5 (run automatically by the notebook's
    "Common stellar grid" cell if the file is absent).
  - `data/` — the repackaged grid lands here and is **git-ignored**.
- **`_figs/`** — generated figures.

## Prerequisites

```bash
pip install cosmos-synthesizer jupytext jupyter
```

Then download Synthesizer's **test** grids (the notebook runs on these — no
production grids needed):

```bash
synthesizer-download --stellar-test-grids --agn-test-grids --dust-grid
```

By default these land in Synthesizer's application-support directory; point the
driver at a custom location with `export SYNTHESIZER_GRID_DIR=/path/to/grids`.

> **Grids.** The test grids are the same physics as the production grids
> (Box: <https://sussex.app.box.com/v/SynthesizerGrids>) at lower resolution —
> the AGN test grids sample each photoionization axis at just two nodes. Because
> both codes read the *same* file, the coarseness never appears as a
> tengri-vs-Synthesizer disagreement. Swapping in a production grid (and
> re-pointing `SYNTHESIZER_GRID_DIR`) is a drop-in higher-resolution re-render.

## Running

```bash
jupytext --to ipynb 01_synthesizer.py
PYTHONPATH=$PWD/../..:$PWD/../../src \
  SYNTHESIZER_GRID_DIR="$HOME/Library/Application Support/Synthesizer/grids" \
  jupyter nbconvert --to notebook --execute --inplace 01_synthesizer.ipynb \
  --ExecutePreprocessor.timeout=900
```

Expected runtime: a few minutes on a CPU. The §8 nebular and §7 panchromatic
panels use tengri's Cue emulator (needs `data/cue_weights.npz`); the first-time
JAX compilation dominates and subsequent runs reuse the persistent cache.

## What the notebook covers

§1 SSPs · §2 delayed-τ SFH · §3 stellar SED · §4 attenuation curves
(Calzetti, power-law) · §5 attenuation applied · §6 dust IR + energy balance
(Draine & Li 2007) · §7 panchromatic · §8 nebular (Synthesizer Cloudy grid vs
tengri Cue) · **§9 AGN — the Unified AGN model** · §12 IGM (Inoue 2014 / Madau 1995).

X-ray and radio are skipped — Synthesizer has no counterpart. The numbering keeps
the gap (§10 X-ray, §11 radio) to line up with the CIGALE master sequence.

### §9 — the Unified AGN model (the focus)

- **§9a disc** — Synthesizer disc vs tengri `kubota_done` (qsosed) and
  `schartmann2005` (broken power law).
- **§9b** disc incident → escaped + transmitted → observed (the disc–line-region
  geometry).
- **§9c NLR** / **§9d BLR** — tengri reads the same Synthesizer Cloudy grids via
  `compute_nlr_sed_synthesizer` / `compute_blr_sed_synthesizer`; line positions
  match (the spectra differ in line representation — narrow Gaussians vs grid bins).
- **§9e torus** — Synthesizer's blackbody torus vs tengri's Nenkova / two-temperature.
- **§9f** — the full `UnifiedAGN` spectrum and the **hard inclination mask**
  (disc + BLR vanish once `inclination + θ_torus > 90°`, the Type-1 → Type-2
  transition).

## What the comparison found

The per-section scalars printed by the notebook are the quantitative record; the
figures in `_figs/` are the visual one. The shared SSP grid (§1) and the
delayed-τ SFH (§2) reproduce Synthesizer to floating point or a fraction of a
percent. The Draine & Li dust IR (§6), with all four knobs matched (qpah, umin,
gamma, alpha), agrees in shape at every band and in energy to 2.6 percent. The
offset reflects different energy-balance conventions: Synthesizer balances on
the reprocessed spectrum (stellar after gas ionization plus nebular, consuming
the Lyman continuum), while tengri balances on incident stellar with LyC
excluded. WISE W1 0.984x is the 3.3 um PAH feature resampled onto each code's
grid. The nebular block (§8) is the deliberate exception — Synthesizer's Cloudy
grid and tengri's Cue emulator use different photoionization inputs, so the Hα
ratio is reported rather than forced to agree.

In §12, Inoue14 IGM matches Synthesizer's Inoue14 to better than 1e-3 at every
redshift (z = 3, 5, 7); Asada25 matches at z ≤ 5. Two documented convention
differences exist: (1) tengri's Madau implementation uses 17 Lyman-series lines
(Prospector table) whereas Synthesizer's Madau96 keeps only the first 4 (Lyα to
Lyδ), so tengri absorbs about twice as much between Lyδ and the Lyman limit, but
only where transmission is already below 0.02 (e.g. z = 5, rest 865–915 Å: tengri
0.0006–0.008 vs Synthesizer 0.0013–0.016); redward of Lyδ the two agree to a few
1e-3. (2) At z = 7 the Asada25 proximate-CGM damping wing is 2.4x stronger in
Synthesizer (its C = 3 λ_α^2 Λ^2 / (8π) omits the Lyα oscillator strength
f_α = 0.4162 that tengri includes; rest 1245 Å: tengri T = 0.976 vs Synthesizer
0.939).

In §9, the disc, torus, and inclination treatment are independent implementations
and compared on shape/amplitude; the line regions share grids and so match in line
content.
