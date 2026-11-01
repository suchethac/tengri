# Nebular Emission: tengri vs BEAGLE

*Date: 2026-04-21. Grid files in `data/neogal/`.*

---

## Overview

BEAGLE (Bayesian Analysis of Galaxies for Physical Inference and Parameter
EStimation; Chevallard & Charlot 2016, MNRAS 462, 1415) uses precomputed
CLOUDY c13.03 photoionization grids as its nebular emission engine. tengri
provides four independent nebular backends for H II regions that cover the same physics with
different trade-offs for speed, flexibility, and differentiability.

---

## BEAGLE's nebular model

### HII region emission (stellar photoionization)

Source: Gutkin, Charlot & Bruzual 2016, MNRAS 462, 1757 (`data/neogal/nebular_emission_Z*.txt`)

CLOUDY c13.03 models of HII regions ionized by the updated Bruzual & Charlot code of
that time (Charlot & Bruzual, in preparation; the CB2013/CB16 templates), single-star,
Chabrier IMF.

Grid axes and coverage:

| Axis | Symbol | Range | N points |
|------|--------|-------|----------|
| Gas metallicity | Z | 0.0001–0.04 (14 files) | 14 |
| Ionization parameter | log U_S | −4.0 to −1.0 | 7 |
| Dust-to-metal ratio | ξ_d | 0.1, 0.3, 0.5 | 3 |
| Hydrogen density | n_H [cm⁻³] | 10, 100, 1000, 10⁴ (10 and 10⁴ present at a sparse subset of nodes; tengri keeps 100 and 1000) | 4 tabulated, 2 kept |
| Carbon-to-oxygen ratio | (C/O)/(C/O)_⊙ | 0.1–1.4 (9 values) | 9 |
| Upper IMF cutoff | m_up [M_⊙] | 100, 300 | 2 |
| **Total** | | | **756 per metallicity file; 10,584 across the 14 files** |

18 emission lines: [OII]3727, Hβ, [OIII]4959/5007, [NII]6548/6584, Hα,
[SII]6717/6731, NV1240, CIV1548/1551, HeII1640, OIII]1661/1666,
[SiIII]1883, SiIII]1888, CIII]1908.

The public tables give line luminosities in L_⊙ (3.826e33 erg/s) per unit SFR (M_⊙/yr)
for a constant star formation rate sustained over 100 Myr. The per-photon templates that
tengri uses are derived from the table's own Hβ through Case B in
`scripts/build_gutkin16_grid.py`.

### AGN NLR emission

Source: Feltre, Charlot & Gutkin 2016, MNRAS 456, 3354 (`data/neogal/AGN_NLR_nebular_feltre16/`)

CLOUDY c13.03 models of AGN narrow-line region gas, ionized by a broken
power-law EUV spectrum f_ν ~ ν^α.

Grid axes:

| Axis | Symbol | Values | N points |
|------|--------|--------|----------|
| Gas metallicity | Z | 0.0001–0.07 (16 files) | 16 |
| Ionization parameter | log U_S | −5.0 to −1.0 | 9 |
| Dust-to-metal ratio | ξ_d | 0.1, 0.3, 0.5 | 3 |
| Hydrogen density | n_H [cm⁻³] | 100, 1000, 10000 | 3 |
| EUV power-law slope | α | −2.0, −1.7, −1.4, −1.2 | 4 |
| **Total** | | | ~5,184 per-metallicity × 16Z = ~83K** |

20 emission lines: adds [OI]6300 vs Gutkin grid; drops mup axis.

---

## tengri nebular backends

### Backend 1: CLOUDY grid — FSPS/Byler+2017

File: `src/tengri/components/nebular/cloudy_grid.py`

CLOUDY photoionization grids computed with BPASS v2.1 binary stellar populations
as the ionizing source (Byler et al. 2017, ApJ 840, 44). **This most directly
parallels BEAGLE's Gutkin+2016 approach** but uses a different ionizing SED
(BPASS vs the Bruzual & Charlot SSPs).

Grid axes: (logZ_gas, log_age, logU) — 3D per SSP grid. The Q_H for normalization
is re-computed from the user's DSPS SSPs at runtime; only the line ratio shape is
fixed to BPASS (documented warning: `CloudyGridIonizingSpectrumWarning`).

**Compared to Gutkin+2016:**
- Fewer emission lines (typically 128 from FSPS grids vs 18 in Gutkin)
- No C/O or n_H axes — less flexibility for abundance ratio science
- Ionizing spectrum: BPASS v2.1 binary (harder) vs the updated Bruzual & Charlot single-star SSPs of Gutkin+2016
- CLOUDY version: c17.01 (or FSPS version) vs c13.03

### Backend 2: Gutkin+2016 NEOGAL (`gutkin16`)

File: `src/tengri/components/nebular/gutkin16.py`; grid `data/gutkin16_grid.h5`, built by
`scripts/build_gutkin16_grid.py` from the public tables in `data/neogal/`.

Per-ionizing-photon line templates for the 18 Gutkin+2016 lines, interpolated with PCHIP over
(logZ_gas, logU, xi_d, log n_H, C/O) with 14, 7, 3, 2 and 9 nodes; the upper mass cutoff
m_up (100 or 300 M_⊙) is chosen at load time. The ionizing photon rate is the user's SSP Q_H,
multiplied by the table's per-photon ratios. The nebular continuum is analytic (free-free plus
two-photon) or absent. Lyα is Case B times the Lyα escape fraction.

**Compared to the BEAGLE model it reproduces:**
- Same lines, same tables, same C/O axis
- The ionizing SED is the user's SSP, not the stellar model of the table
- n_H is kept at 100 and 1000 cm⁻³; the 10 and 10⁴ cm⁻³ columns are not in the grid
- No N/O axis

### Backend 3: CB_19 (3MdBs database)

File: `src/tengri/components/nebular/cloudy_cb19.py`

2,358,330 CLOUDY c17.01 models from Martinez-Paredes et al. 2023 (arXiv:2308.05604).
Ionizing SED: CB19 SSPs, a later generation of the same Bruzual & Charlot code that Gutkin+2016 used.

Grid axes: (log O/H, log age, logU, log n_H, log C/O, ΔN/O, HbFrac)

**Compared to Gutkin+2016:**
- More grid points (2.36M vs ~10.6K for the Gutkin+2016 tables) with higher sampling density
- CLOUDY c17.01 vs c13.03 — updated atomic data, recombination coefficients
- Adds N/O axis (Gutkin+2016 fixes N/O at scaled-solar)
- Adds age axis (explicitly tracks evolution of ionizing spectrum with stellar age)
- Adds HbFrac axis (matter-bounded nebulae and ionizing photon escape)
- Same C/O axis as Gutkin+2016
- Direct parallel to BEAGLE's HII grid with additional dimensions

### Backend 4: Cue neural-network emulator

File: `src/tengri/components/nebular/cue.py`

Neural-network emulator for nebular emission (Li et al. 2025, ApJ 986, 9;
arXiv:2405.04598). 12 input parameters: 7 ionizing spectrum shape
parameters (4 spectral indices + 3 log-luminosity ratios) plus 5 gas parameters
(logU, logZ, log n_H, ΔN/O, ΔC/O). Predicts line luminosities for ~271 lines.

**Compared to Gutkin+2016:**
- ~15× more emission lines predicted
- Arbitrary ionizing spectrum shape via 7 parameters — **not fixed to the Bruzual & Charlot or BPASS SSPs**
- Accepts AGN power-law ionizing spectrum directly (agn_nebular.py)
- JAX-native, JIT-compatible, suitable for VI and HMC
- No age axis: Cue was trained for time-averaged ionizing spectra; tengri handles
  this by passing the DSPS mass-averaged ionizing spectrum to the Cue input
- Metallicity range: similar to Gutkin+2016 (Z ~ 0.0001–0.04)

### AGN NLR backend: Cue + analytic

File: `src/tengri/components/nebular/agn_nebular.py`

For AGN-ionized gas tengri provides:

1. **Cue (default)**: `agn_nlr_cue()` converts the AGN EUV power-law slope α to
   the 7 Cue ionizing spectrum parameters via `agn_ionspec_from_alpha_pl()`, then
   calls Cue. **Advantage over Feltre+2016**: arbitrary ionizing spectrum shape
   variation is possible; JAX-native and differentiable.

2. **Feltre grid (optional)**: `FeltreNLRBackend` loads the Feltre+2016 grids
   directly (the same data BEAGLE uses). Note: the raw Feltre+2016 ASCII files
   are in `data/neogal/AGN_NLR_nebular_feltre16/`. A build script
   (`scripts/download_feltre_grid.py`) documents how to convert them to HDF5.
   Interpolation: triweight for continuous axes (logU, logZ, logn_H), nearest-
   neighbor for discrete axes (α, ξ_d).

---

## Synthesizer (CLOUDY c23.01) grids

Files: `data/synthesizer_grids/test_grid_agn-nlr.hdf5`, `test_grid_agn-blr.hdf5`

Downloaded via `synthesizer-download --agn-test-grids`. These are test (2-point)
grids; production grids available at the Sussex Box repository.

Grid axes (6D): BH mass, accretion rate (Eddington), cos(inclination),
metallicity, ionization parameter, hydrogen density.

| Axis | Notes |
|------|-------|
| `mass` (BH mass) | Unlike BEAGLE/tengri which use L_bol or L_acc |
| `accretion_rate_eddington` | Eddington ratio |
| `cosine_inclination` | Geometry: torus opening angle encoded differently |
| `metallicities` | |
| `ionisation_parameter` | |
| `hydrogen_density` | |

**215 emission lines** (vs 18–20 in Gutkin/Feltre). CLOUDY c23.01 — most up-to-date
atomic data of any grid available.

**Key structural difference from tengri:** Synthesizer uses BH mass + Eddington
ratio as primary axes. tengri uses L_bol directly (`agn_log_lbol`). Both
parameterize the same physics, but L_bol is more directly observable and has
fewer degeneracies in SED fitting.

---

## Side-by-side comparison

| Feature | BEAGLE (Gutkin+2016) | BEAGLE (Feltre+2016) | tengri gutkin16 | tengri CB_19 | tengri Cue | tengri Feltre |
|---------|---------------------|---------------------| ------------- |-------------|-----------|--------------|
| Use case | HII regions | AGN NLR | HII regions | HII regions | HII + AGN | AGN NLR |
| CLOUDY version | c13.03 | c13.03 | c13.03 | c17.01 | (trained) | c13.03 |
| N emission lines | 18 | 20 | 18 | ~18 (ratio-based) | ~271 | 20 |
| logU range | −4 to −1 | −5 to −1 | −4 to −1 | included | included | −5 to −1 |
| Metallicity range | 0.0001–0.04 | 0.0001–0.07 | 0.0001–0.04 | wider (3MdB) | ~same | 0.0001–0.07 |
| n_H axis | 4 values (10, 100, 1000, 10⁴) | 3 points | 2 points (100, 1000) | included | included | 3 points |
| C/O axis | 9 points | — | 9 points | included | included | — |
| N/O axis | — | — | — | included | included | — |
| Ionizing SED | Bruzual & Charlot SSPs of that time (fixed) | power-law α | user's SSP Q_H with the table's per-photon ratios | CB19 | arbitrary | power-law α |
| JAX / JIT | no | no | yes | yes | yes | yes |
| VI/HMC compatible | no | no | not checked | yes | yes | yes |
| Differentiable | no | no | yes (PCHIP) | yes (triweight) | yes (NN) | yes (triweight) |

---

## Key differences from BEAGLE

1. **Ionizing spectrum flexibility**: BEAGLE fixes the ionizing SED to the stellar
   SSP grid (the Bruzual & Charlot SSPs). tengri's Cue backend accepts any ionizing spectrum shape,
   enabling physically consistent line predictions when the ionizing source differs
   from the SSP (e.g., stripped stars, AGN contribution to HII region ionization).

2. **Differentiability**: BEAGLE uses grid interpolation that is not differentiable
   through the photoionization model. tengri's Cue backend is a smooth neural network,
   enabling gradient-based inference (VI, HMC) on nebular parameters jointly with
   SFH and dust parameters.

3. **C/O and N/O jointly**: BEAGLE's Gutkin+2016 grid has C/O as an explicit axis
   but fixes N/O to scaled-solar. tengri's CB_19 backend has both. Neither BEAGLE
   nor tengri-Cue have both simultaneously (Cue handles them as offsets).

4. **AGN NLR**: BEAGLE uses Feltre+2016 grids. tengri defaults to Cue (arbitrary
   ionizing spectrum, more lines, faster) with Feltre+2016 as an optional backend
   using the identical underlying grids.

5. **Emission line completeness**: BEAGLE predicts 18–20 key diagnostic lines.
   tengri-Cue predicts ~271 lines, enabling wider wavelength coverage including
   UV lines critical for high-z JWST spectroscopy.

---

## Grid files on disk

```
data/neogal/
  nebular_emission_Z*.txt          # Gutkin+2016 HII grids (14 Z files, ~4 MB)
data/gutkin16_grid.h5              # packaged HDF5 built by scripts/build_gutkin16_grid.py
  AGN_NLR_nebular_feltre16/
    nlr_nebular_Z*.txt             # Feltre+2016 AGN NLR grids (16 Z files, ~381 KB)
data/synthesizer_grids/
  test_grid_agn-nlr.hdf5           # Synthesizer NLR test grid (2-pt, CLOUDY c23.01)
  test_grid_agn-blr.hdf5           # Synthesizer BLR test grid (2-pt, CLOUDY c23.01)
```

The Synthesizer production NLR/BLR grids (HDF5, hundreds of MB) are available
from the Sussex Box repository; use `synthesizer-download` (present in `.venv`)
to download them when needed.

---

## References

- Chevallard & Charlot 2016, MNRAS 462, 1415 — BEAGLE
- Gutkin, Charlot & Bruzual 2016, MNRAS 462, 1757 — HII grids
- Feltre, Charlot & Gutkin 2016, MNRAS 456, 3354 — AGN NLR grids
- Byler et al. 2017, ApJ 840, 44 — FSPS/CLOUDY nebular grids
- Martinez-Paredes et al. 2023, arXiv:2308.05604 — CB_19 / 3MdBs
- Li et al. 2025, ApJ 986, 9 — Cue emulator (arXiv:2405.04598)
- Lovell et al. 2025, Open J. Astrophys., arXiv:2508.03888 — Synthesizer
