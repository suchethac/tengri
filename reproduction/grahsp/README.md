# GRAHSP ↔ tengri

A component-by-component comparison of tengri's GRAHSP AGN model with upstream GRAHSP's own
spectra. The reference is Buchner et al. 2024, A&A 692, A161
(arXiv:2405.19297, doi:10.1051/0004-6361/202449372).

## What this notebook compares

| Component | tengri | Upstream spectrum in the reference |
|---|---|---|
| Big blue bump (smooth bending power law) | `grahsp_sbpl` | yes, 38 sets |
| Broad and narrow emission lines | `grahsp` (blr, nlr) | yes, 38 sets (type-1 broad lines) |
| FeII forest | `grahsp` (feii) | yes, 38 sets |
| Log-Gaussian torus and Si feature | `grahsp` (torus) | yes, 38 sets |
| Balmer continuum | `grahsp` (feii side) | yes, the `abc` sets |
| Netzer disc (`grahsp_netzer`) | not in the reference | no; stated from the paper relation |
| MN12 template torus (`grahsp_mn12`) | not in the reference | no; stated from the paper relation |

The notebook prints the installed registry's GRAHSP block names rather than relying on this
table. The reference file holds the 38 parameter sets of `tests/crossval/test_grahsp_vs_upstream_reference.py`.

## Prerequisites

None beyond the repository. The reference file `data/grahsp_upstream_reference.h5` is
committed, and the driver reads only that file and tengri's own templates
(`data/grahsp/grahsp_templates.h5`). Upstream GRAHSP is never imported at render time.

To regenerate the reference, run `scripts/build_grahsp_reference.py` with a GRAHSP checkout and
its Python environment. Regeneration is a build-time job and is not needed to render this page.

## Running

```bash
python scripts/render_reproduction_notebook.py grahsp
```

This runs the recipe in `reproduction/CONTRACT.md` §7, stamps the render, and publishes it to
`docs/reproduction/`. Figures are written to `_figs/grahsp_*.png`.

## Scope

The component comparisons use the public GRAHSP component function `evaluate_grahsp_agn` at
matched parameters, the same function the crossval test uses. The full composite in section 4
is computed the same way, from the same components. The Netzer disc and MN12 torus have no
upstream spectrum in the reference, so their normalization differences are stated from the
paper relation (section 3.5) and are not tested against upstream output.

## References

- Buchner, J., Starck, H., Salvato, M., et al. 2024, A&A 692, A161, "Genuine Retrieval of the AGN
  Host Stellar Population (GRAHSP)", arXiv:2405.19297.
