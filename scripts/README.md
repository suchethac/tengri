# `scripts/` — Internal Reproducibility Tooling

These scripts are **not user tutorials.** They are developer/researcher utilities
used to build template grids, benchmark inference engines, generate paper
figures, and run reproducibility checks.

If you're new to `tengri`, start here instead:

- **`README.md`** at the repo root — installation and quick-start
- **`notebooks/`** — annotated Jupyter tutorials (Jupytext `.py` source)
- **`examples/`** — Sphinx-Gallery example scripts
- **`docs/`** — full documentation (also at the project's docs site)

## Categories

| Prefix             | Purpose                                                       |
|--------------------|---------------------------------------------------------------|
| `build_*`          | Construct template grids (Silva04 torus, CAT3D, RELAGN, ...)  |
| `convert_*`        | Convert upstream data products to tengri's HDF5 schema        |
| `download_*`       | Fetch external data (filter sets, SSPs, dust templates)       |
| `benchmark_*`      | Performance microbenchmarks (forward model, inference, JIT)   |
| `bench_*`          | Same — alternate naming                                       |
| `profile_*`        | JAX profiling / memory diagnostics                            |
| `diagnose_*`       | Targeted diagnostics for specific components or regressions   |
| `generate_*`       | Generate plots, mock data, or derived outputs                 |
| `verify_*`, `test_*` | Standalone smoke tests (not part of the pytest suite)       |
| `sync_*`           | Notebook ↔ docs sync utilities (Jupytext)                     |
| `_*` (underscore)  | Helpers; not intended to be invoked directly                  |

## Requirements

These scripts assume the **dev install**:

```bash
pip install -e ".[dev]"
```

Many scripts also expect data files under `data/` that are **not** shipped with
the package — see the [SSP grid setup](../README.md#installing-an-ssp-grid)
section of the main README, plus per-script docstrings for any extra
prerequisites.

## Caveats

- Scripts may write large output files (HDF5 grids, profiling traces, PNG
  galleries) and are not guarded against overwriting existing artifacts.
- Some scripts hard-code GPU/CPU assumptions or experimental flags. Read the
  module docstring before running.
- Output paths default to repository-relative locations; override via the
  documented env vars (e.g. `TENGRI_PAPER_FIG_DIR`) where available.
