# Hensley & Draine 2023 Astrodust+PAH — usage and parameter reference

This directory contains example scripts reproducing every figure in
[`brandonshensley/Astrodust/notebooks/model_file_tutorial.ipynb`](https://github.com/brandonshensley/Astrodust/blob/main/notebooks/model_file_tutorial.ipynb)
using tengri's `DustEmissionSEDComponent`. The data are loaded from the
canonical Harvard Dataverse FITS file [doi:10.7910/DVN/3B6E6S][1] —
"Astrodust+PAH Model Output" — repacked into `data/astrodust_templates.h5`
by `scripts/build_astrodust_hdf5.py`.

[1]: https://doi.org/10.7910/DVN/3B6E6S

## Build

```
python scripts/build_astrodust_hdf5.py --output data/astrodust_templates.h5 --download
```

Downloads the 3 MB FITS file from Harvard Dataverse and writes a
~1 MB HDF5 grid.

## Run all examples

```
for f in examples/astrodust_hd23/0*.py; do python "$f"; done
```

Each script writes a PNG next to itself.

## Examples

| Script | Reproduces | What it shows |
|---|---|---|
| `01_size_distribution.py`        | tutorial fig 1 | Per-H grain volume `(4π/3)a³ dn/dlna/n_H` for Astrodust + PAHs |
| `02_emission_vs_lgU.py`          | tutorial fig 7 | `λI_λ/N_H/U` per H per U at several log U |
| `03_components_at_fiducial_U.py` | tutorial fig 6 | Astrodust vs PAH vs spinning dust at fiducial `log U = 0.2` |
| `04_sedmodel_dust_emission_swap.py` | n/a | MBB ↔ Draine+2021 PAHspec ↔ HD23 Astrodust template swap |
| `05_ionization_alignment.py`     | tutorial fig 2 | `f_ion(a)` and `f_align(a)` |
| `06_extinction_and_scattering.py`| tutorial figs 3-5 | `τ_λ/N_H`, polarized extinction, and albedo |
| `07_spinning_dust.py`            | tutorial fig 9 | Spinning dust `I_ν/N_H` at 10-100 GHz |
| `08_polarized_emission.py`       | tutorial fig 8 | Polarized emission `λP_λ/N_H` and Astrodust polarization fraction |

## Parameters: what the model exposes

The H&D 2023 published file [3B6E6S] is a single configuration that bakes
in all 17+ underlying parameters at the H&D 2022 fiducial values. tengri
exposes them in three layers depending on what's actually fittable.

### Layer 1 — Free parameter (continuous, fittable today)

| Name | Range | Meaning | Source |
|---|---|---|---|
| `dust_lgU` | -3.0 to +6.0 | log₁₀ of the starlight intensity scaling factor `U` (where `U=1` is the local Galactic ISRF as parameterised by Mathis-Mezger-Panagia 1983). The model interpolates linearly in `lgU` across the 91-point published grid (step 0.1). | HDU 7 axis |

### Layer 2 — Categorical config (set at component construction, frozen)

These live on `DustEmissionSEDComponentConfig` when `template="astrodust"`.

| Field | Choices | Default | Meaning |
|---|---|---|---|
| `astrodust_component`              | `"total"` / `"astrodust"` / `"pah"` | `"total"` | Which thermal-emission column from HDU 7 to use. `"total"` = Astrodust continuum + PAHs (sum); `"astrodust"` = continuum only; `"pah"` = PAH features only. |
| `astrodust_include_spinning_dust`  | `True` / `False`                    | `False`   | Add the (mostly U-independent) microwave spinning-dust spectrum (HDU 9) on top of the thermal IR. Important if your SED fit extends to ~10-100 GHz (Planck/AME bands). |
| `astrodust_f_cnm`                  | float in [0, 1]                     | `0.28`    | Cold-Neutral-Medium filling factor used to mix CNM + WNM spinning-dust spectra. Published default is 0.28. Could in principle be made fittable in a future version. |
| `astrodust_template_path`          | path or `None`                      | `None`    | Override the location of `astrodust_templates.h5`. `None` = check `TENGRI_ASTRODUST_PATH` env var, then fall back to `data/astrodust_templates.h5`. |

### Layer 3 — Frozen at H&D 2022 fiducial (would require per-grain cross-section dataset)

These parameters are **inferable in principle** but require the separate
`doi:10.7910/DVN/PEXRD0` per-grain cross-section dataset and a JAX
implementation of the size-distribution integration (the
`changing_size_distribution_tutorial.ipynb` notebook from the upstream
repo shows the integration in pure NumPy). **Not currently exposed in
tengri.** The fiducial values from H&D 2022 Table 1 are listed here so
you know what's frozen:

#### PAH size distribution (`size_car`)

| Symbol | Value | Meaning |
|---|---|---|
| `B1`        | `7.52 × 10⁻⁷`  | Amplitude of the small-PAH log-Gaussian (carrier of 3.3 μm feature). |
| `B2`        | `8.09 × 10⁻¹⁰` | Amplitude of the large-PAH log-Gaussian. |
| `a01`       | `4.0 Å`        | Peak radius of the small-PAH log-Gaussian. |
| `a02`       | `30 Å`         | Peak radius of the large-PAH log-Gaussian. |
| `σ1`, `σ2`  | `0.40, 0.40`   | log-widths of the two PAH log-Gaussians. |
| `amin_PAH`  | `4.0 Å`        | Minimum PAH grain radius (physical floor). |

#### Astrodust size distribution (`size_Ad`)

| Symbol | Value | Meaning |
|---|---|---|
| `BAd`       | `3.31 × 10⁻¹⁰` | Amplitude of the log-normal Astrodust component. |
| `a0_Ad`     | `63.8 Å`       | Peak radius of the log-normal. |
| `σ_Ad`      | `0.353`        | log-width of the log-normal. |
| `A0..A5`    | `(2.97e-5, -3.40, -0.807, 0.157, 7.96e-3, -1.68e-3)` | Polynomial coefficients of the second Astrodust component (eq. 18 of H&D 2022). |
| `amin_Ad`   | `4.5 Å`        | Minimum Astrodust grain radius (silicate photolysis cutoff). |

#### PAH ionization function

| Symbol | Value | Meaning |
|---|---|---|
| `a_h` | `10 Å` | Transition radius for `f_ion(a) = 1 - 1/(1 + a/a_h)`. |

#### Grain alignment (only affects polarization, not total emission)

| Symbol | Value | Meaning |
|---|---|---|
| `a_align`     | `0.0749 μm` | Transition radius for `f_align(a) = f_max / (1 + (a_align/a)^α_align)`. |
| `α_align`     | `1.80`      | Power-law index of the alignment function. |
| `f_max`       | `1.00`      | Maximum alignment efficiency. |

## Composition with other dust IR templates

`DustEmissionSEDComponent` switches between three IR-emission templates
behind one config field:

```python
from tengri.components.dust.emission_component import (
    DustEmissionSEDComponent,
    DustEmissionSEDComponentConfig,
)

# Modified blackbody (analytic 2-param)
mbb = DustEmissionSEDComponent()

# Draine+2021 PAHspec (categorical starlight + lgU)
pahspec = DustEmissionSEDComponent(
    config=DustEmissionSEDComponentConfig(
        template="draine2021_pah",
        pahspec_starlight="auto",
        pahspec_auto_age_myr=10.0,
        pahspec_auto_log_z_solar=0.0,
        pahspec_auto_sps_family="BC03",
    ),
)

# Hensley & Draine 2023 Astrodust+PAH (lgU only; finer 91-point grid)
astrodust = DustEmissionSEDComponent(
    config=DustEmissionSEDComponentConfig(
        template="astrodust",
        astrodust_component="total",
        astrodust_include_spinning_dust=True,  # 10-100 GHz coverage
    ),
)
```

All three implement the same `precompute → apply` Protocol and reuse the
same energy-balance loop with `state.derived["L_ir"]`.

## Citation

Hensley & Draine 2023, ApJ 948, 55, [arXiv:2208.12365](https://arxiv.org/abs/2208.12365). DOI [10.3847/1538-4357/acc4c2](https://doi.org/10.3847/1538-4357/acc4c2).
