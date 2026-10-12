# GRAHSP ↔ tengri

A component-by-component comparison of tengri's GRAHSP AGN blocks with upstream GRAHSP's own
spectra. The reference is Buchner et al. 2024, A&A 692, A161
(arXiv:2405.19297, doi:10.1051/0004-6361/202449372).

## What this notebook compares

The reference holds 38 parameter sets. The page uses six of them: the fiducial, two Balmer-continuum
sets (`abc_0_p3`, `abc_1_p0`), two torus-width sets (`coolwidth_0_p35`, `coolwidth_0_p55`), and the
type-2 set. The Netzer and MN12 comparisons use the fiducial parameters. The attenuation comparison
uses two E(B-V) sets (`ebv_0_p1`, `ebv_agn_0_p2`).

| Component | tengri block | Upstream spectrum in the reference |
|---|---|---|
| Big blue bump (smooth bending power law) | `grahsp_sbpl` | yes (fiducial and sets above) |
| Broad and narrow emission lines | `grahsp` (blr, nlr) | yes: integrals and centroids; widths for single-line windows |
| FeII forest | `grahsp` (feii) | yes: integral over 200–800 nm |
| Log-Gaussian torus and Si feature | `grahsp` (torus) | yes (torus-width sets and fiducial) |
| Balmer continuum | `grahsp` (blr) | yes (the two `abc` sets) |
| Netzer disc | `grahsp_netzer` | yes, from the `netzer_mn12` builder variant (fiducial); compared by the source table's header label |
| MN12 template torus | `grahsp_mn12` | yes, from the `netzer_mn12` builder variant (fiducial, absolute units) |
| Attenuation (biattenuation) | `grahsp_biatten` | yes (two E(B-V) sets) |
| Type-2 composite | `grahsp` (nlr only) | yes (type-2 set) |

Upstream's importer assigns the Netzer disc table's 16 columns to (log M, spin, mdot) in its own
enumeration order rather than the order of the table's header, so only 2 of its 16 labels match
the header. The disc upstream labels (8, 0, 0.3) is the header column `m80_a0_mdot003`, i.e.
(8, 0, 0.03). The page compares tengri's (8, 0, 0.03) node with that spectrum, and a check cell
decodes the header to show the mapping.

The notebook prints the installed registry's GRAHSP block names rather than relying on this
table. Upstream's emission lines are rendered at nine nodes per line and interpolated linearly,
so upstream's line widths are about 22% larger than the Gaussian they state; the page reports this
and does not correct tengri to match it.

## Prerequisites

None beyond the repository. The reference file `data/grahsp_upstream_reference.h5` is
committed. The SED model's stellar grid is downloaded on first render with `tengri.download_ssp`
(`fsps_mist_miles_chabrier`). Upstream GRAHSP is never imported at render time.

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
upstream is interpolated linearly in wavelength onto that grid. The crossval test
(`tests/crossval/test_grahsp_vs_upstream_reference.py`) compares the component functions at
upstream's own nodes for all 38 sets, which is the finer test.

## References

- Buchner, J., Starck, H., Salvato, M., et al. 2024, A&A 692, A161, "Genuine Retrieval of the AGN
  Host Stellar Population (GRAHSP)", arXiv:2405.19297.
- Grandi, S. A. 1982, ApJ 255, 25, "The 3000 Å bump in quasars", doi:10.1086/159799.
