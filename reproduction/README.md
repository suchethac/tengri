# Physics reproduction studies

Component-by-component physics reproductions: tengri against the reference
SED-fitting codes the community already relies on. Each subfolder
holds one comparison.

```
reproduction/
├── cigale/                     # Boquien et al. 2019, A&A 622, A103
├── bagpipes/                   # Carnall et al. 2018, MNRAS 480, 4379
├── prospector/                 # Johnson et al. 2021, ApJS 254, 22
├── agnfitter/                  # Martínez-Ramírez et al. 2024, A&A 688, A46
├── prospect_r/                 # Robotham et al. 2020, MNRAS 495, 905
└── synthesizer/                # Lovell et al. 2025 (OJA) + Roper et al. 2026 (JOSS) — cite both
```

Each comparison ships a notebook (`01_<code>.py`, jupytext percent
format), thin code-specific driver modules under `_drivers/`, and the
rendered figures. All six folders are complete; the Synthesizer
notebook is kept out of the published docs for now. AGNFITTER-RX is the AGN-first member of the series — a
radio-to-X-ray deep dive on the four accretion-disk and four torus
libraries — and reads the external code's template libraries directly
rather than running its fitter. The Synthesizer notebook focuses on the
Unified AGN model (disc, NLR, BLR, torus, and the inclination
geometry), with tengri reading the same Synthesizer Cloudy AGN grids
for the line regions. ProSpect, the R-based GAMA code, is driven live
from its notebook through `rpy2`.

## Conventions

Every notebook follows [CONTRACT.md](CONTRACT.md): layout, shared
helpers, figure naming, the full-SED capstone, rendering rules, and how
the published copies under `docs/reproduction/` stay in sync with the
sources here. Read it before adding a comparison or re-rendering an
existing one.

## Running a notebook

```bash
cd reproduction/<code>
jupytext --to ipynb 01_<code>.py
jupyter nbconvert --to html --execute 01_<code>.ipynb
```

Prospector additionally needs `SPS_HOME` pointed at an FSPS checkout.
See the per-code README inside each subfolder for prerequisites and
data setup specific to that comparison. Re-rendering for the docs site
follows [CONTRACT.md](CONTRACT.md) §7–8.
