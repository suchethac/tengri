# CB_19 Nebular Grid (Charlot & Bruzual 2019)

The `CB19Backend` implements the Charlot & Bruzual 2019 CLOUDY photoionization
grid from the 3MdB_17 database (Martinez-Paredes et al. 2023, arXiv:2308.05604).
It is an alternative to `CloudyGridBackend` offering wider abundance coverage
(C/O ratio, N/O offset) and explicit matter-bounded nebulae via the HbFrac axis.

:::{note}
The CB_19 grid contains only **emission line ratios** — no nebular continuum.
See [No nebular continuum](#no-nebular-continuum) below for what this means and
how to work around it.
:::

## Status (2026-09)

`scripts/download_cb19_templates.py` cannot currently build
`data/cb19_templates.h5`. A read-only probe of the 3MdB servers on
2026-09-07 found no `ref='CB_19'` row, under any spelling, in any reachable
database:

- `3MdB_17.tab_17` (the table this script queries) carries six refs:
  `PNe_2020`, `PNe_2021`, `BOND_2`, `CALIFA_2`, `BOND`, `CALIFA`.
- `3MdB.tab` carries eight: `PNe_2014`, `CALIFA`, `PNe_2014_c13`, `DIG_HR`,
  `PNe_2016`, `BOND`, `CALIFA_ah`, `HII_CHIm`.
- `3MdBs.projects` carries four: `Allen08`, `Gutkin16`, `Alarie19s`,
  `Allen08-cut`.

No `CB_19`, or any near-miss spelling (`CB19`, `CB_2019`, `Bruzual`,
`Charlot`), appears in any of the three.

The 3MdB project page for CB_19
(<https://sites.google.com/site/mexicanmillionmodels/the-different-projects/cb_19>)
carries a standing notice from the grid's own authors: there is a bug in how
chemical abundances and metallicities are defined in the grid, and they will
produce a new grid and an erratum explaining the consequences. That page
also names the database as `3MdB`, not `3MdB_17` as the script queries; a
direct query of `3MdB.tab` finds no CB_19 rows either, so the mismatch is
not (solely) which of the two databases is queried.

`data/README.md` forbids redistributing the CB_19 grid as part of tengri, and
no permissively licensed repackaging of this exact product is known to
exist, so tengri does not ship one.

**What this means today.** `neb={'type': 'cb19'}` still works against a
grid you supply yourself: place a real CB_19 HDF5 file with genuine
variation at `data/cb19_templates.h5` (or under `$TENGRI_DATA_DIR`) --
`SEDModel.build`'s `neb` grammar has no `grid` key for `cb19` (unlike
`cloudy`, `mappings`, and `mappings_agn`), so the resolved default path is
the only route through the build API; `CB19Backend(grid_path=...)` accepts
an explicit path directly if you construct the backend yourself. Until 3MdB
republishes the grid, `scripts/download_cb19_templates.py` cannot populate
that file, and the alternatives are `neb={'type': 'cue'}`,
`neb={'type': 'cloudy', 'grid': <path>}`, or `neb={'type': 'ssp'}` with a
wNE SSP grid.

## Quick start

```python
from tengri import SEDModel, Fixed

# Default: SSP ionizing spectrum, Kroupa IMF, radiation-bounded.
# Requires data/cb19_templates.h5 (built once — see below).
model = SEDModel.build(
    ssp_data=ssp, observation=obs,
    neb={
        'type': 'cb19',
        'logU':     Fixed(-3.0),
        'logZ_gas': Fixed(0.0),      # log10(Z_gas/Zsun); 0.0 = solar
        'log_nH':   Fixed(2.0),      # log10 cm⁻³
        'co':       Fixed(-0.36),    # log10(C/O), solar
        'dno':      Fixed(0.0),      # ΔN/O offset (log dex)
    },
)
```

Short keys inside the ``neb`` group resolve to the full parameter names
(``logU`` → ``neb_logU``); passing the full names works too. Note
``logZ_gas`` is **log10(Z/Zsun)** — relative to solar, so ``0.0`` means
solar gas metallicity.

The template file was built once, before first use, with:

```bash
python scripts/download_cb19_templates.py
```

which downloads CLOUDY models from the 3MdB_17 database (`3mdb.astro.unam.mx`,
table `tab_17`, ref='CB_19') and saves `data/cb19_templates.h5`. As described
in [Status (2026-09)](#status-2026-09) above, this route does not currently
work: supply your own `data/cb19_templates.h5` until 3MdB republishes the
grid.

## Unit convention: Hβ ratios → L/Q_H

**This is the single most important design detail to understand.**

CB_19 stores all line fluxes as dimensionless ratios relative to Hβ:

$$\text{stored value} = \frac{L_\text{line}}{L_{H\beta}}$$

The tengri SED pipeline requires each line luminosity normalized by the
ionizing photon rate Q_H (in Lsun per photon s⁻¹), so it can be weighted
by the ionizing photon output of each SSP age bin:

$$\frac{L_\text{line}}{Q_H} = \frac{L_\text{line}}{L_{H\beta}} \times \frac{L_{H\beta}}{Q_H}$$

The Case B conversion factor (Osterbrock & Ferland 2006, Table 4.4;
T_e = 10⁴ K, n_e = 10² cm⁻³; also eq. 1 of Byler et al. 2017, ApJ 840 44) is:

$$\frac{L_{H\beta}}{Q_H} = 4.78 \times 10^{-13}\ \text{erg photon}^{-1}
= \frac{4.78 \times 10^{-13}}{3.828 \times 10^{33}}\ L_\odot\,\text{s\,photon}^{-1}
\approx 1.249 \times 10^{-46}\ L_\odot\,\text{s\,photon}^{-1}$$

This constant (`_HB_PER_QH_LSUN`) is stored both in `cloudy_cb19.py` and in the
HDF5 root attrs (`hb_per_qh_lsun`) for reproducibility.

The conversion is applied inside `predict_nebular_line_luminosities` for every
SSP age bin before summing:

```
L_line(age_i) = ratio(age_i) × _HB_PER_QH_LSUN × Q_H(age_i) × weight_i × (1 − f_esc)
```

### Why not store L/Q_H directly in the HDF5?

The ratio space is more compact and portable — other codes using the 3MdB_17
database can ingest the ratios directly without knowing tengri's unit system.
The conversion is a single scalar multiply and is numerically exact.

## No nebular continuum

Unlike `CloudyGridBackend` (which includes free-bound and two-photon continuum),
**CB_19 stores only line ratios**. The `predict_nebular_continuum()` method
returns zeros.

This is by design at the source: 3MdB_17 does not store continuum-level outputs
for the CB_19 run. The grid is optimized for line-ratio diagnostics (BPT, O3N2,
abundance calibrations) where the continuum is not needed.

**Consequence:** `CB19Backend` contributes no nebular continuum to the SED.
For science cases that need nebular continuum (e.g., Balmer continuum emission,
Lyman-break region), stack `CB19Backend` on top of `CloudyGridBackend`:

```python
# Stack: use Cue/CloudyGrid continuum + CB_19 lines
from tengri.components.nebular import CloudyGridBackend, CB19Backend

cloudy = CloudyGridBackend("data/cloudy_grid_mist.h5", ssp_data)
cb19  = CB19Backend()

# In a custom forward pass:
wave_cont, lum_cont = cloudy.predict_nebular_continuum(...)
lum_lines           = cb19.predict_nebular_line_luminosities(...)
```

For most SED-fitting applications (photometry + optical spectroscopy at z > 0.05
where the Lyman series is redshifted out of the observed window), the continuum
contribution is negligible and `CB19Backend` alone is sufficient.

## Grid axes and ranges

The 6D interpolation grid (all axes interpolated linearly via
`jax.scipy.ndimage.map_coordinates`):

| Parameter | Parameter name | Grid range | Default |
|-----------|--------------|------------|---------|
| log(O/H) | `neb_logZ_gas` (converted) | −5.06 → −2.58 | 0.0 (solar; log Z/Zsun) |
| log age / yr | (SSP age bins) | 6.0 → 10.6 | — |
| log U | `neb_logU` | −4.0 → −1.0 | −3.0 |
| log n_H / cm⁻³ | `neb_log_nH` | 1 → 4 | 2.0 |
| log(C/O) | `neb_co` | −1.0 → 0.15 | −0.36 (solar) |
| ΔN/O (log dex) | `neb_dno` | −0.25 → 0.25 | 0.0 |

**HbFrac** (7th axis, discrete): sets the matter-bounded escape fraction.
HbFrac = L_Hβ(matter-bounded) / L_Hβ(radiation-bounded); HbFrac = 1.0 is
fully radiation-bounded. Snapped to the nearest grid point at `CB19Backend`
init — not interpolated. Grid points: [0.0, 0.1, 0.25, 0.5, 0.75, 1.0].

### Metallicity axis detail

CB_19 uses the CLOUDY c17.01 solar oxygen abundance
12 + log(O/H)_⊙ = 8.93 (log(O/H)_⊙ ≈ −3.07). Internally tengri carries
absolute log₁₀(Z) with log₁₀(Z_⊙) = −1.848 (Asplund et al. 2009); the
user-facing ``neb_logZ_gas`` is log₁₀(Z/Z_⊙), and the translation layer
adds the −1.848 offset before the value reaches the backend — so the
``neb_logZ_gas`` in the conversion below is the internal absolute value.

The internal conversion is:

```
log_OH = neb_logZ_gas + _LOG_OH_OFFSET
_LOG_OH_OFFSET = −3.07 − (−1.848) = −1.222
```

This assumes O/H scales proportionally with total metallicity Z (solar
abundance ratios). The 7 grid log(O/H) values correspond to:

| 12+log(O/H) | log(O/H) | ≈ Z/Z_⊙ |
|-------------|----------|---------|
| 6.87 | −5.06 | 0.009 |
| 7.87 | −4.06 | 0.09  |
| 8.48 | −3.45 | 0.48  |
| 8.73 | −3.20 | 0.75  |
| 8.81 | −3.12 | 0.87  |
| 9.07 | −2.86 | 1.41  |
| 9.35 | −2.58 | 2.14  |

## HbFrac: matter-bounded nebulae

HbFrac parameterizes density-bounded (matter-bounded) nebulae where ionizing
photons can escape before the Strömgren sphere closes. It is defined as:

$$\text{HbFrac} = \frac{L_{H\beta}(\text{matter-bounded})}{L_{H\beta}(\text{radiation-bounded})}$$

- **HbFrac = 1.0** (default): fully radiation-bounded, all ionizing photons
  absorbed within the nebula. Standard assumption for star-forming galaxies.
- **HbFrac < 1.0**: matter-bounded, with Lyman continuum escape fraction
  ≈ 1 − HbFrac. Relevant for Lyman continuum emitters and high-ionization
  compact HII regions.

The HbFrac axis is **not interpolated** — it is snapped to the nearest grid
point at init time, as intermediate values have no physical interpretation
within the CLOUDY model grid.

## SSP vs CSF ionizing spectra

`CB19Backend` supports two ionizing SED types via the `sed_type` argument:

- **`"SSP"`** (default): single stellar population (instantaneous burst).
  Grid axis: log_age from 6.0 to 10.6 yr (41 points).
- **`"CSF"`**: continuous star formation (constant SFR). Grid axis:
  log_age from 6.0 to 9.9 yr (24 points).

For the standard tengri SED pipeline (which builds a full SFH from weighted
SSP contributions), use `sed_type="SSP"`.

```python
# CSF mode — for simple models with a single continuous SFR
backend = CB19Backend(sed_type="CSF")
```

## Reference and citation

If you use `CB19Backend` in published work, please cite:

- **Martinez-Paredes et al. 2023** (arXiv:2308.05604) — the CB_19 grid paper
- **Charlot & Bruzual 2019** (in prep / private comm.) — the ionizing SSP SEDs
- **Osterbrock & Ferland 2006**, "Astrophysics of Gaseous Nebulae and Active
  Galactic Nuclei", 2nd ed., Table 4.4 — the Hβ/Q_H conversion
- **Byler et al. 2017**, ApJ 840 44 — consistent use of the same L_Hβ/Q_H factor
- **Morisset et al. 2015**, A&A 579, A89 — the 3MdB(s) database framework

## API reference

```{eval-rst}
.. autoclass:: tengri.components.nebular.CB19Backend
   :members:
   :undoc-members:
```
