# Fitting workflows

The fitting notebooks in the spine cover photometry, spectroscopy, joint
data, and the degeneracies that come with each.

| Notebook | Topic |
|----------|-------|
| [`05_fitting_photometry`](../spine/05_fitting_photometry) | Photometric fit end to end, with proper convergence diagnostics |
| [`06_fitting_spectroscopy`](../spine/06_fitting_spectroscopy) | Optical spectroscopy: Lick indices, line masking, calibration polynomial |
| [`07_joint_photo_spec`](../spine/07_joint_photo_spec) | Joint photo + spec to break age/dust/metallicity degeneracies |
| [`08_emission_lines`](../spine/08_emission_lines) | Emission-line diagnostics (BPT, Hα-SFR consistency) |
| [`10_fastspecfit_joint_fit`](../spine/10_fastspecfit_joint_fit) | Joint fit: DESI photometry + emission-line fluxes, with measured fit times |
| [`11_catalog_fits`](../spine/11_catalog_fits) | A whole catalog fit in parallel: LSST+Euclid free-redshift photo-z, timed end to end |

For batch fitting many galaxies with shared infrastructure, see
[advanced/batch_fitting](../advanced/batch_fitting).

```{toctree}
:maxdepth: 1
:hidden:

../advanced/batch_fitting
../advanced/mappings_photo
```
