# Fitting a catalog of spectra in one batch

Use the batched spectroscopic stack when many galaxies share one instrument setup and one
model family, and differ only in their data: pixel grid, redshift, resolution, masks and
photometry. The stack compiles the forward model once per bucket of galaxies and evaluates
every galaxy in the bucket through that compiled function.

Use the per-galaxy `Fitter` when the model differs between galaxies: a different parameter
set, a different line catalog, a Gaussian-process star formation history, or spectral indices.
The per-galaxy path also remains the reference the batched path is tested against.

## When to use it

| Situation | Path |
|---|---|
| Many spectra, one instrument, one model family, free redshift | Batched stack |
| Different model per galaxy, or a model feature the batch does not evaluate | Per-galaxy `Fitter` |
| A handful of spectra with a fixed redshift each | Per-galaxy `Fitter` |

## The workflow

The stack has four steps. Each galaxy is an observation record, the template is one
`SEDModel` that fixes everything the galaxies share, and the fits run on buckets of records.

```python
import numpy as np
import jax.numpy as jnp
from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Spectroscopy,
    Uniform,
    make_agebins_from_zred,
)
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.line_list import LineList

wave = np.linspace(4000.0, 7500.0, 600)
resolution = 2000.0 + 3000.0 * (wave - 3600.0) / (9800.0 - 3600.0)
spectroscopy = Spectroscopy(
    wave_obs=jnp.asarray(wave),
    resolution=resolution,
    resolution_matrix=gaussian_resolution_bands(jnp.asarray(wave), resolution, 11),
    resample="conserving",
    eline_mode="marginalized",
    eline_catalog=LineList.default_optical(),
    eline_prior_sigma=1e-27,
    eline_prior_type="flat",
    eline_broad=False,
)
template = SEDModel.build(
    ssp_data=ssp,
    observation=Observation(
        spectroscopy=spectroscopy,
        photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]),
    ),
    sfh={
        "type": "continuity",
        "all_params": FREE,
        "bin_edges_gyr": [float(e) for e in make_agebins_from_zred(zred=0.2)],
        "log_total_mass": Uniform(6.0, 10.5),
    },
    dust_attenuation={
        "type": "two_component",
        "law": "calzetti",
        "tau_bc": Uniform(0.0, 2.0),
        "tau_diff": Uniform(0.0, 1.5),
        "other_params": Fixed(DEFAULT),
    },
    met={"logzsol": Uniform(-1.5, 0.2), "all_params": Fixed(DEFAULT)},
    neb={"type": "none"},
    redshift=Uniform(0.0, 0.2),
)
```

<!-- no-exec -->
```python
from tengri.observation.batched import GalaxySpectrum, build_spectro_batches
from tengri.inference.batched_spectro import fit_spectra_map_vmap, fit_spectra_catalog_mcmc

records = [
    GalaxySpectrum(
        wave=w_i, flux=f_i, ivar=iv_i, resolution=bands_i, z=z_i,
        mask=m_i, phot_flux=pf_i, phot_err=pe_i,
    )
    for (w_i, f_i, iv_i, bands_i, z_i, m_i, pf_i, pe_i) in catalog
]
batches = build_spectro_batches(records, quantum=256, model=template)
map_fit = fit_spectra_map_vmap(template, batches, n_steps=300)
posterior = fit_spectra_catalog_mcmc(template, batches, n_warmup=200, n_samples=200)
```

1. **Records.** `GalaxySpectrum` holds one galaxy: the observed grid `wave` [Angstrom],
   `flux` and `ivar` in F_nu [erg/s/cm^2/Hz], the resolution as a banded matrix (the
   `resolution` field also accepts a scalar or per-pixel R), the redshift `z`, an optional pixel
   `mask`, and optional photometry `phot_flux` and `phot_err`. Pixels with a non-finite or
   non-positive `ivar` drop out. The template must use the same kind of resolution as the records.
2. **Template.** One `SEDModel` with a free redshift fixes the model family, the parameter
   set, the line catalog, the filter union and the SFH ladder. `Fixed(z)` is refused, because the
   batch carries each galaxy's redshift.
3. **Buckets.** `build_spectro_batches(records, quantum=..., model=template)` pads each pixel
   count up to a multiple of `quantum`, groups galaxies with the same static specification, and
   stacks each group into one batch.
4. **Fits.** `fit_spectra_map_vmap` runs a MAP optimization for every galaxy in a bucket with one
   Adam scan. `fit_spectra_catalog_mcmc` samples the posterior of every galaxy with the catalog
   engine. Treat MAP as a warm start for the sampler. For a non-parametric star formation history,
   MAP is not an estimate of the posterior median (see [Measured numbers](#measured-numbers)).

## What each galaxy carries, and what the template fixes

Each galaxy supplies its own data, traced through the compiled function as arrays. A batch holds
these 20 leaves, in this order, each with a leading galaxy axis:

| Leaf | Content |
|---|---|
| `wave` | Observed grid [Angstrom], padded to the bucket's pixel count |
| `pix_mask` | 1.0 for a good real pixel, 0.0 for padded, masked or zero-weight pixels |
| `flux` | Observed F_nu [erg/s/cm^2/Hz]; 0 on excluded pixels |
| `sigma` | 1-sigma flux error; 1 on excluded pixels |
| `r_data` | Banded resolution rows; empty when the resolution is Gaussian |
| `resolution_pp` | Per-pixel spectral resolution R; empty when the resolution is banded |
| `z` | Redshift |
| `cal_lo`, `cal_hi` | Smallest and largest real wavelength [Angstrom], the calibration range |
| `phot_flux`, `phot_err` | Photometry and its errors [erg/s/cm^2/Hz]; empty without photometry |
| `phot_presence` | 1.0 where a filter has a finite flux and a positive error |
| `sfh_bin_edges_gyr` | Per-galaxy SFH bin edges [Gyr]; empty when the template ladder is used |
| `line_mask` | 1.0 for each catalog line that lands on the galaxy's real grid at its redshift |
| `eline_resolution` | Spectral resolution R that sets the emission-line profile width |
| `ebmv_mw` | Milky Way E(B-V) of the foreground, per galaxy |
| `chol_data` | Bands of the lower Cholesky factor of the galaxy's pixel covariance; empty without one |
| `segment` | Calibration segment id of each pixel; zeros without segments |
| `cal_scale_center`, `cal_scale_sigma` | `log10_center` and `sigma_dex` of the galaxy's log-normal prior on the mean calibration multiplier |

The template fixes everything that changes a shape or a branch of the compiled function: the
SED family and its parameter set, the emission-line catalog, the filter union, the SFH type and
the number of its edges, the resolution form, the number of calibration segments, the covariance
offsets, and `quantum`. A change to any of these is a new bucket and a new compile.

## Choosing `quantum`

`quantum` sets how many distinct padded pixel counts the stack creates. Each distinct padded
count is one bucket, and each bucket compiles once. Larger `quantum` means fewer buckets and more
padding, so more pixels evaluated per galaxy.

For 24 galaxies with fitted pixel counts from 4,159 to 4,751. Padding is the padded pixels minus the fitted pixels, divided by the fitted pixels summed over the 24 galaxies:

| `quantum` | Buckets | Padded counts (`n_max`) | Padding |
|---|---|---|---|
| 256 | 3 | 4,352, 4,608, 4,864 | 3.0% |
| 4864 | 1 | 4,864 | 9.2% |

One bucket means `quantum` is at least the span of the pixel counts. Choose the value by how many
compiles you can afford against how much padding you can accept. The padded pixels carry zero
weight and do not change the likelihood of any real pixel.

## Emission lines and calibration

**Emission lines.** `eline_mode` sets how the catalog lines enter the likelihood. `"fitted"`
samples each amplitude. `"marginalized"` with `eline_prior_type="flat"` integrates the amplitudes
out analytically, so no line parameters are sampled. The marginal is exact for a Gaussian
amplitude prior and a linear design. Use `"marginalized"` for catalog-wide fits where the line
amplitudes are nuisance parameters.

**Model-centered line prior.** `Spectroscopy(eline_mode="marginalized", eline_prior_type="model")`
replaces the flat prior with a Gaussian on each line amplitude, centered on the model's own
prediction for that line. The prediction is carried through the same projection as the continuum,
including the dust and the Milky Way foreground. The width is `eline_prior_width_dex`, in dex
(default 0.3). The parameter `eline_aperture_frac` scales the prior mean by the fraction of the
line flux that falls inside the fiber. It is `Fixed(1.0)` by default, and
`SEDModel.build(eline_aperture_frac=...)` frees it or sets it to a prior. The prior works with the
lines marginalized alone and inside the joint marginal with the calibration polynomial. A line
outside a galaxy's grid or under a pixel mask contributes nothing. Use it instead of the flat prior
when the lines should follow the model's dust and star formation history, so that the Balmer
decrement constrains the attenuation.

**Lines outside a galaxy's grid.** A catalog line that falls outside a galaxy's observed grid at
its redshift contributes nothing to that galaxy. Its column is removed from the design, so it
neither adds to the model nor changes the marginal. The `line_mask` leaf of each batch records
which lines are in the grid. Its row sums give the number of lines each galaxy carries. In the
measured set below, a 3.3 to 3.9 thousand-angstrom window with sky, camera-join and Ca K masks held
12 to 24 of the 39 catalog lines.

**Balmer series.** `LineList.default_optical()` carries H8 to H14, He I 3889 and [Ne III] 3968
beside the lower-order lines. With `neb={"type": "cue", "balmer_series_n_max": 50}`, the default,
Cue's emission appends H11 to H50 at the Case-B ratios of Storey and Hummer (1995) to H-beta.
`balmer_series_n_max=None` leaves the series at Cue's own last line.

**Calibration.** `calibration_marginalize=True` with `cal_n_poly=3` integrates a Chebyshev
calibration polynomial out of the spectral term, with the prior width `cal_prior_sigma`. The
joint marginal of calibration and lines is exact, so the two marginals are not applied one after
the other. The calibration marginal requires `calibration_order=0` in the template, and the
combination with fitted lines (`eline_mode="fitted"`) is refused. Use `"marginalized"` lines
with it.

**Per-segment calibration scales.** For a spectrum made of several camera segments, give the
template `Spectroscopy(calibration_segments=...)`, either the pixel count of each segment or one
integer id per pixel, and each segment gets its own free scale in the calibration marginal. A
galaxy supplies its own ids with `GalaxySpectrum(segment_ids=...)`. Galaxies with different segment
counts form separate buckets. `cal_prior="scale_free"` replaces the Gaussian prior on the
coefficients with a prior flat in data-flux units; the default is `"gaussian"`.

**Fiber-fraction prior.** `Spectroscopy(cal_scale_prior=(log10_center, sigma_dex))` puts a
log-normal prior on the mean calibration multiplier q over a window, and q enters the fit as one
sampled latent, `cal_log_q`. `cal_scale_window=(lo, hi)` [Angstrom] sets the window; the default
is the segment that holds the r band's effective wavelength. `GalaxySpectrum(cal_scale_prior=...)`
sets the prior galaxy by galaxy.

## Correlated noise

`GalaxySpectrum(covariance=BandedMatrix)` gives a galaxy a symmetric banded pixel covariance
[(erg/s/cm^2/Hz)^2]. Every galaxy that carries one uses the same diagonal offsets, and the
template is built without a `Spectroscopy.covariance` of its own. The spectral term, the
calibration marginal and the line marginals whiten the residual by one banded Cholesky factor.
Masked and padded pixels are identity rows of that factor, so they contribute nothing.
`ivar` still selects the good pixels.

## Star formation history and stellar grids

**Young-fraction coordinates.** `sfh={"type": "continuity", "coordinates": "young_fraction"}`
samples the mass fractions of the continuity SFH from the youngest bin outward, each
Uniform(0, 1). Use it when the photometry pins the young mass: the posterior then has its own
axis for that mass. The default, `"ratios"`, samples the log SFR ratios.

**[alpha/Fe] grids.** A 4-D SSP grid with an [alpha/Fe] axis is contracted over metallicity at each
alpha node before the alpha interpolation, so a sampled `met_alpha_fe` adds no per-draw copy of
the cube. A free AGB dust-shell weight (`agb_dust_weight`) collapses alpha first and runs the
rest of the stellar sum on a 3-D cube.

## Choosing a sampler

`fit_spectra_catalog_mcmc` takes `sampler="nuts"` (the default), `"hmc"`, `"chees"` or `"ghmc"`.
Use `"chees"` for catalog-wide posteriors. `"ghmc"` runs MEADS with `n_chains=32` and
`n_superchains=8` by default and a fixed-length warmup, and it returns the per-galaxy
`nested_rhat`. On narrow posteriors it does not reach NUTS: at DESI-like signal to noise the
measured gap is 0.48 posterior widths.

## Measured numbers

The numbers were measured at commit `a608d682f`. This page describes the code at `fb0aa11c0`.

The numbers come from 12 mock spectra fit two ways on one machine: the per-galaxy `Fitter` on a
`Fixed(z)` build of the template, and the batched stack on the free-redshift template. The mocks
have 560, 600 or 640 pixels over 4000 to 7500 angstrom, banded resolution R = 300, a 5 percent
calibration tilt and three injected emission lines (H beta, [O III] 5007 and H alpha). The model is
a DPL star formation history with the mass free, `eline_mode="marginalized"`, and
`calibration_marginalize=True` with `cal_n_poly=3`. The batched path used `quantum=64`, which gives
two buckets: 4 galaxies at 576 padded pixels and 8 at 640.

The machine is an Apple M4 Pro with 14 cores and 48 GiB of memory, on CPU at float64.

**Per-galaxy path.** Times are summed over the galaxies of each pixel count, four each.

| Stage | 560 pixels | 600 pixels | 640 pixels | All 12 |
|---|---|---|---|---|
| `SEDModel.build` | 0.27 s | 0.36 s | 0.51 s | 1.13 s |
| `Fitter` construction | 2.49 s | 2.23 s | 2.36 s | 7.08 s |
| First `neg_log_posterior` evaluation (compile) | 1.64 s | 2.47 s | 2.43 s | 6.53 s |
| 300 Adam MAP steps | 3.10 s | 4.36 s | 3.49 s | 10.95 s |

**Batched path.** All 12 galaxies in one call per step.

| Stage | Wall | JIT cache misses |
|---|---|---|
| `build_spectro_batches(quantum=64)` | 0.05 s | not counted |
| `fit_spectra_map_vmap`, first call, 300 steps | 22.72 s | 573 |
| `fit_spectra_map_vmap`, second call, 300 steps | 0.82 s | 2 |

The jit-miss counter counts build-time compiles too. The second call logged 2 cache misses at the
same step count; their cause was not traced.

**Posterior check.** Three galaxies, with `n_warmup=100` and `n_samples=200`. The batched run is
`fit_spectra_catalog_mcmc` on the free-redshift template. The `Fitter` run is `mcmc_nuts` on the
galaxy's `Fixed(z)` build. The table gives the median of log10 M and the 16 to 84 percent width.

| Galaxy | Pixels | z | Truth | Batched median (width) | `Fitter` median (width) | Difference / `Fitter` width | Wall, batched / `Fitter` |
|---|---|---|---|---|---|---|---|
| 0 | 560 | 0.050 | 9.00 | 9.0024 (0.0020) | 9.0023 (0.0021) | 0.07 | 11.1 s / 11.2 s |
| 4 | 600 | 0.086 | 10.00 | 9.9996 (0.0020) | 9.9996 (0.0017) | 0.02 | 12.1 s / 11.2 s |
| 11 | 640 | 0.150 | 9.25 | 9.2517 (0.0014) | 9.2518 (0.0015) | 0.02 | 12.4 s / 9.1 s |

With 200 draws the median of each run has a Monte Carlo error of about 0.04 widths, so the difference
of two medians has about 0.06 widths. The largest difference, 0.07 widths on galaxy 0, is within
that error.

**Peak memory.** Each build ran in a fresh process. The table gives the peak RSS of one build on the
100-pixel synthetic grid. Most of that figure is the process baseline: about 494 MB after importing the
package and the grid, so the build itself added about 11 MB.

| Build (synthetic grid) | Peak RSS after build | Build wall |
|---|---|---|
| Free-redshift template, process-baseline figure | 504.8 MB | 0.19 s |
| `Fixed(z)` build, galaxy 0, process-baseline figure | 508.3 MB | 0.14 s |

The build size on a real grid is the figure to plan memory with. On the real grid of the 24-galaxy run,
the free-redshift template built at a peak of 632 MB and one `Fixed(z)` build at 754 MB. That model has
no dust emission and no IGM term. The larger build peak of models with those terms is tracked in
[#3153](https://github.com/suchethac/tengri/issues/3153).

**Larger set.** A measurement on 24 galaxies at 4,159 to 4,751 fitted pixels, on the same machine, gave a first batched MAP call of 45 s including compile and 127 s
for 300 steps (5.3 s per galaxy). A per-galaxy `Fitter` NUTS run on one galaxy with a dense mass matrix
ran for more than 14.5 minutes and 2.6 GB before a memory kill. That timing used the `Fitter` defaults at
ten free parameters and is not comparable to the table above.

What these numbers show:

- Batched MAP over 12 galaxies costs about the same as the whole per-galaxy path on its first call,
  22.7 s against 25.7 s including compile. Once compiled, the 300-step batched call costs 0.82 s.
- The batched and per-galaxy posteriors agree on the three sampled galaxies within 0.07 widths.
- After 300 Adam steps the per-galaxy MAP estimates for the three mocks with truth 9.00 sit 0.048 to
  0.049 dex high, while the batched MAP estimates are within 0.002 dex of truth on all 12 mocks.
- For a non-parametric star formation history, use `fit_spectra_catalog_mcmc` for the posterior.

After 300 Adam steps the two paths' MAP values can differ by 0.05 dex on the same mock, while their
posterior medians agree within Monte Carlo error. The MAP is therefore a warm start for the sampler,
not an estimate.

## Float32

Float32 is a supported mode. The stack's tests check that the float32 gradients are finite and
nonzero at log10 M = 4.0, with the data scaled to the working range. The per-galaxy spectroscopic
forward itself is not finite in pure float32 at every mass. On the fixture of
[#3149](https://github.com/suchethac/tengri/issues/3149) it is non-finite above log10 M of about 5.25.
Check the fit output for finite values before using a float32 catalog fit.

## What the stack does not do

Each limit below names what to do instead.

- **A template covariance.** Give each `GalaxySpectrum` its own `covariance=BandedMatrix` and
  build the template without one.
- **Student-t noise and a calibration floor.** The batched likelihood is Gaussian with no floor.
  Use Gaussian noise.
- **The Cloudy line prior and broad lines.** Use `eline_prior_type="flat"` or `"model"` and
  `eline_broad=False`, or `eline_mode="fitted"` or `"off"`.
- **Spectral indices.** Use the per-galaxy `Fitter`.
- **Gaussian-process star formation histories.** Use a parametric history with the batch, or the
  per-galaxy `Fitter` for a Gaussian-process history.
- **Line-flux and line-ratio channels.** Remove them from the template, or use the per-galaxy
  `Fitter`.
