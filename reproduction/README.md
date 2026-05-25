# Reproduction studies

This folder contains component-by-component reproductions of tengri against established SED-fitting codes. Each subfolder corresponds to one reference code and demonstrates that tengri reproduces its physics surface in a differentiable JAX implementation.

## Structure

```
reproduction/
├── cigale/
│   ├── 01_cigale.py           # Main CIGALE ↔ tengri reproduction notebook
│   ├── _drivers/              # Thin wrappers for CIGALE's pcigale modules
│   │   ├── units.py           # Unit conversions (CIGALE W/nm ↔ tengri erg/s/Hz)
│   │   ├── cigale_driver.py   # pcigale module instantiation and SED extraction
│   │   ├── cigale_ssp_to_dsps.py  # One-off: BC03 templates → DSPS HDF5
│   │   └── data/
│   │       └── bc03_from_cigale.h5  # CIGALE BC03 grids in DSPS schema
│   └── _figs/                 # Output figure directory
├── bagpipes/                  # (future) Bagpipes reproduction
├── prospector/                # (future) Prospector reproduction
└── synthesizer/               # (future) Synthesizer reproduction
```

## Running the notebooks

### CIGALE reproduction

```bash
# Convert notebook to ipynb (jupytext)
cd reproduction
jupytext --to ipynb cigale/01_cigale.py

# Execute in Jupyter or render to HTML
jupyter nbconvert --to html cigale/01_cigale.ipynb
```

Prerequisites:
- `tengri` (git repo available; see parent `..`)
- `pcigale` (`pip install pcigale`)
- `jupytext` (`pip install jupytext`)

### Regenerating BC03 templates

The `bc03_from_cigale.h5` file is a one-off port of CIGALE's BC03 library to the DSPS HDF5 schema. To regenerate:

```bash
python cigale/_drivers/cigale_ssp_to_dsps.py \
  --input /path/to/cigale/data/bc03/  \
  --output cigale/_drivers/data/bc03_from_cigale.h5
```

This reads CIGALE's FITS files and writes an HDF5 file that both `pcigale` and `tengri.load_ssp()` can consume identically.

## Rendering to docs

To include a reproduction notebook in the main documentation:

```bash
jupytext --to ipynb reproduction/cigale/01_cigale.py
jupyter nbconvert --to html --execute reproduction/cigale/01_cigale.ipynb --output-dir docs/_build/html/
```

Then add a toctree entry in `docs/index.md` under a new "Reproduction" group.

## Design notes

- **No tengri driver**: The notebook calls tengri's public API directly (`tengri.SEDModel.build()`, `tengri.load_ssp()`, etc.) so readers see exactly what they would write.
- **CIGALE driver is minimal**: `_drivers/cigale_driver.py` wraps pcigale so that cells stay short; it is not a general-purpose interface.
- **Same wavelength grids**: All comparisons use the same rest-frame wavelength grid and units (erg/s/Hz on Angstroms).
- **Component-by-component**: Each figure isolates one physics layer (SSP, SFH, nebular, dust, AGN, etc.) to pinpoint agreements and gaps.
