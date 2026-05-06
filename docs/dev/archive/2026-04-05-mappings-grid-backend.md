# Session Plan: 2026-04-05 — MAPPINGS III + V Full Grid Backend

## Motivation

The existing `shock.py` implements only a thin slice of Allen+2008 MAPPINGS data: 8 velocity
points, 10 optical lines, solar abundance, n=1 cm⁻³, no magnetic field, no
precursor/shock separation. The full MAPPINGS library is a 4D grid over
(v_shock, B/√n, abundance, density) with 40+ lines UV–IR.

**References read this session:**
- Allen et al. 2008, ApJS 178 20 — MAPPINGS III (arXiv:0805.0204)
- Sutherland & Dopita 2017, ApJS 229 34 — MAPPINGS V (arXiv:1702.07453)
- Alarie & Morisset 2019, RMxAA 55 279 — 3MdBs database (arXiv:1908.08579)

---

## Architecture

The shock term is already an additive layer in the pipeline:

```
SED = stellar_attenuated + nebular_backend + shock + dust_IR + AGN
```

This upgrade is entirely internal to `shock.py`. No changes to `sed_pipeline.py` or
nebular backends.

---

## Data Sources

| Version | Source | Format |
|---------|--------|--------|
| MAPPINGS III | https://cds.unistra.fr/~allen/mappings_page1 | ASCII (wavelength-ordered flux ratios) |
| MAPPINGS V | https://zenodo.org/records/14140949 (3MdBs, Alarie+2019) | CSV (14 files, ~4132 lines) |

**Preferred V source:** 3MdBs Zenodo CSVs — already parsed by parameter, 3 component
types (shock/precursor/combined), standard column format. Key files:
`shock_fluxes.csv`, `precursor_fluxes.csv`, `shockprecursor_fluxes.csv`, `*_propts.csv`.

---

## HDF5 Layout (`data/mappings_templates.h5`)

```
mappings_templates.h5
├── mappings3/                      # Allen+2008 MAPPINGS III
│   ├── attrs: source, doi, velocity_range
│   ├── velocities_kms              # (N_v,)
│   ├── b_over_sqrt_n_uG_cm3_2      # (N_B,)
│   ├── log_density_cm3             # (N_n,) — log10(n)
│   ├── abundance_names             # (5,) — solar, 2xsolar, dopita2005, lmc, smc
│   ├── line_names                  # (N_lines,) — PyNeb format e.g. "HA_6563A"
│   ├── line_wavelengths_aa         # (N_lines,) — rest-frame vacuum
│   ├── shock_ratios                # (N_abund, N_n, N_v, N_B, N_lines) rel. to Hβ
│   ├── precursor_ratios            # same shape
│   ├── combined_ratios             # same shape
│   └── hbeta_log_lum_erg_s        # (N_abund, N_n, N_v, N_B) log10 erg/cm²/s
│
└── mappings5/                      # Alarie+2019 3MdBs (MAPPINGS V)
    ├── attrs: source, doi, zenodo_record
    ├── velocities_kms              # (N_v,) — 200–1000 in 25 km/s steps
    ├── b_field_uG                  # (N_B,) — [1e-3, 1e-2, 1e-1, 1, 10, 100]
    ├── log_density_cm3             # (N_n,)
    ├── abundance_names             # string array
    ├── line_names                  # (N_lines,) ~40 key diagnostic lines
    ├── line_wavelengths_aa         # (N_lines,) vacuum
    ├── shock_ratios                # relative to Hβ
    ├── precursor_ratios
    ├── combined_ratios
    └── hbeta_log_lum_erg_s
```

Only ~40 key diagnostic lines from MAPPINGS V are stored in HDF5 (not all 4132) to
keep the interpolation tensors tractable. Full 4132-line list available from raw Zenodo CSVs.

---

## Files Modified / Created

| File | Change |
|------|--------|
| `scripts/download_mappings_templates.py` | **NEW** — download + preprocess both grids → HDF5 |
| `src/tengri/models/nebular/shock.py` | **MAJOR** — HDF5 grid loading + multi-axis interpolation |
| `src/tengri/core/param_spec.py` | **MINOR** — add `shock_b_over_sqrt_n`, `shock_abundance`, `shock_component`, `shock_version`; un-reserve `shock_log_density` |
| `tests/unit/test_shock.py` | **NEW** — 9 regression tests |

---

## New `shock_line_ratios` Signature

```python
def shock_line_ratios(
    shock_velocity: float,              # km/s: 100-1000 (III) or 200-1000 (V)
    shock_log_density: float = 0.0,    # log10(n/cm⁻³) — now active
    shock_b_over_sqrt_n: float = 1.0,  # μG cm^(3/2), nearest grid point
    shock_abundance: str = "solar",    # solar|2xsolar|dopita2005|lmc|smc
    shock_component: str = "combined", # shock|precursor|combined
    shock_version: str = "mappings5",  # mappings3|mappings5
) -> dict[str, float]
```

**Interpolation:** velocity → `jnp.interp` (continuous); B-field + density →
`jnp.searchsorted` nearest-neighbor (log-spaced discrete); abundance + component +
version → Python string index (static).

**Fallback:** if `data/mappings_templates.h5` missing, falls back to current hardcoded
Table 5 arrays with `DeprecationWarning`. Same pattern as dust templates.

---

## New `_SHOCK_PARAMS` Entries

```python
"shock_log_density":    Fixed(0.0),       # was "reserved" — now active
"shock_b_over_sqrt_n":  Fixed(1.0),       # μG cm^(3/2); equipartition ~3.23
"shock_abundance":      Fixed("solar"),   # categorical
"shock_component":      Fixed("combined"), # categorical
"shock_version":        Fixed("mappings5"), # mappings3|mappings5
```

---

## Verification

```bash
python scripts/download_mappings_templates.py   # build data/mappings_templates.h5
pytest tests/unit/test_shock.py -v             # 9 tests
pytest tests/ -q                               # full suite, no regressions
ruff check src/ tests/ && ruff format --check src/ tests/
```
