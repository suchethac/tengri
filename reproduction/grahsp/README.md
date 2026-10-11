# GRAHSP ↔ tengri

A component-by-component comparison of tengri's GRAHSP AGN blocks with upstream GRAHSP's own
spectra. The reference is Buchner et al. 2024, A&A 692, A161
(arXiv:2405.19297, doi:10.1051/0004-6361/202449372).

## What this notebook compares

| Component | tengri block | Upstream spectrum in the reference |
|---|---|---|
| Big blue bump (smooth bending power law) | `grahsp_sbpl` | yes, 38 sets |
| Broad and narrow emission lines | `grahsp` (blr, nlr) | yes, 38 sets (integrals, centroids, widths) |
| FeII forest | `grahsp` (feii) | yes, 38 sets |
| Log-Gaussian torus and Si feature | `grahsp` (torus) | yes, 38 sets |
| Balmer continuum | `grahsp` (blr) | yes, the `abc` sets |
| Netzer disc | `grahsp_netzer` | yes, from the `netzer_mn12` builder variant |
| MN12 template torus | `grahsp_mn12` | yes, from the `netzer_mn12` builder variant |

The notebook prints the installed registry's GRAHSP block names rather than relying on this
table.

## Prerequisites

None beyond the repository. The reference file `data/grahsp_upstream_reference.h5` is
committed, and the SED model's stellar grid is the committed `data/ssp_prsc_miles_chabrier_wNE...`
file. Upstream GRAHSP is never imported at render time.

To regenerate the reference, run `scripts/build_grahsp_reference.py` with a GRAHSP checkout and
its Python environment. Regeneration is a build-time job and is not needed to render this page.

## Running

```bash
python scripts/render_reproduction_notebook.py grahsp
```

This runs the recipe in `reproduction/CONTRACT.md` §7, stamps the render, and publishes it to
`docs/reproduction/`. Figures are written to `_figs/grahsp_*.png`.

## Scope

tengri is evaluated only through the public API: `SEDModel.build` with the composable `agn={...}`
grammar, then `model.predict`. The comparison is made on the model's own wavelength grid, and
upstream is interpolated linearly in wavelength onto that grid. Emission lines are compared by
integrated flux, centroid and width, because the model grid does not resolve every profile.
The crossval test (`tests/crossval/test_grahsp_vs_upstream_reference.py`) compares the component
functions at upstream's own nodes, which is the finer test.

## References

- Buchner, J., Starck, H., Salvato, M., et al. 2024, A&A 692, A161, "Genuine Retrieval of the AGN
  Host Stellar Population (GRAHSP)", arXiv:2405.19297.
