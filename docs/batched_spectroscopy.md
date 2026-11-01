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

Each galaxy supplies its own data, traced through the compiled function as arrays:

- the observed grid, padded to the bucket's pixel count, and the pixel mask;
- the flux, the inverse variance, and the banded resolution operator;
- the redshift;
- the SFH bin edges in Gyr, when the galaxy has its own ladder (`sfh_bin_edges_gyr`);
- the calibration range, the photometry and its presence mask;
- the line mask: which catalog lines land on the galaxy's observed grid at its redshift;
- the line resolution, and the Milky Way E(B-V) of the foreground, when the template has one.

The template fixes everything that changes a shape or a branch of the compiled function: the
SED family and its parameter set, the emission-line catalog, the filter union, the SFH type and
the number of its edges, the resolution form, and `quantum`. A change to any of these is a new
bucket and a new compile.

## Choosing `quantum`

`quantum` sets how many distinct padded pixel counts the stack creates. Each distinct padded
count is one bucket, and each bucket compiles once. Larger `quantum` means fewer buckets and more
padding, so more pixels evaluated per galaxy.

For 24 galaxies with fitted pixel counts from 4,159 to 4,751:

| `quantum` | Buckets | Padded counts (`n_max`) | Padding |
|---|---|---|---|
| 256 | 3 | 4,352, 4,608, 4,864 | 3.0% of the fitted pixels, per galaxy on average |
| 4864 | 1 | 4,864 | 9.4% of the fitted pixels, per galaxy on average |

One bucket means `quantum` is at least the span of the pixel counts. Choose the value by how many
compiles you can afford against how much padding you can accept. The padded pixels carry zero
weight and do not change the likelihood of any real pixel.

## Emission lines and calibration

**Emission lines.** `eline_mode` sets how the catalog lines enter the likelihood. `"fitted"`
samples each amplitude. `"marginalized"` with `eline_prior_type="flat"` integrates the amplitudes
out analytically, so no line parameters are sampled. The marginal is exact for a Gaussian
amplitude prior and a linear design. Use `"marginalized"` for catalog-wide fits where the line
amplitudes are nuisance parameters.

**Lines outside a galaxy's grid.** A catalog line that falls outside a galaxy's observed grid at
its redshift contributes nothing to that galaxy. Its column is removed from the design, so it
neither adds to the model nor changes the marginal. The `line_mask` leaf of each batch records
which lines are in the grid. Its row sums give the number of lines each galaxy carries. In the
measured set below, a 3.3 to 3.9 thousand-angstrom window with sky, camera-join and Ca K masks held
12 to 24 of the 39 catalog lines.

**Calibration.** `calibration_marginalize=True` with `cal_n_poly=3` integrates a Chebyshev
calibration polynomial out of the spectral term, with the prior width `cal_prior_sigma`. The
joint marginal of calibration and lines is exact, so the two marginals are not applied one after
the other. The calibration marginal requires `calibration_order=0` in the template, and the
combination with fitted lines (`eline_mode="fitted"`) is refused. Use `"marginalized"` lines
with it.

## Measured numbers

The numbers below come from a set of 24 mock DESI-like dwarf spectra, fit with the template and
records of the workflow above. They ran on an Apple M4 Pro (14 cores, 48 GiB),
CPU only, at float64, on the batched stack at commit `a608d682f`.

| Quantity | Measurement |
|---|---|
| Template build, free redshift | 3.8 s, peak RSS 632 MB |
| One per-galaxy build, `Fixed(z)` | 0.72 s, peak RSS 754 MB |
| Batched MAP, 24 galaxies, one call, first call | 45 s, including compile |
| Batched MAP, 24 galaxies, 300 Adam steps | 127 s, 5.3 s per galaxy |
| Batched MAP vs truth, log10 M | mean +0.19 dex, rms 0.26 dex |
| Batched ChEES, one galaxy | 242 s including compile, 0 divergences |
| Batched ChEES median vs truth | within 0.25 posterior widths |
| Per-galaxy `Fitter` NUTS, one galaxy | more than 14.5 min and 2.6 GB before a memory kill |

Read these as measurements on this machine under load, not as benchmarks of the method. The
jit-miss counter counts build-time compiles too: 874 misses on the first MAP call. The fit program
itself compiles once per bucket and recompiles only when the step count changes, which added 4
misses at a new scan length.

Two conclusions follow for the recommended path:

- MAP under-converges a continuity star formation history at 300 steps. Its mass is biased by
  +0.19 dex. Use `fit_spectra_catalog_mcmc` for the posterior.
- The per-galaxy NUTS fit at 10 free parameters used a dense mass matrix and was stopped by the
  memory limit. The batched timing used ChEES on one galaxy, so the two timings are not like for like.

The build memory above is for a model without dust emission or an IGM term. Models with those
terms have a larger build peak, tracked in [#3153](https://github.com/suchethac/tengri/issues/3153).

The per-galaxy MAP time for the same 24 galaxies was not measured on this set, so the speed
comparison above is per-galaxy NUTS against batched ChEES on one galaxy.

## Float32

Float32 is supported as a mode. The mass window in which the spectroscopic forward stays
finite in pure float32 is not yet established. The reproducer in
[#3149](https://github.com/suchethac/tengri/issues/3149) gives non-finite output above
log10 M of about 5.25 on its fixture. Use float64 for catalog fits until that issue closes.

## What the stack does not do

Each limit below names what to do instead.

- **Banded pixel covariance.** The batched likelihood uses diagonal noise. For correlated noise,
  use the per-galaxy `Fitter` until [#3133](https://github.com/suchethac/tengri/issues/3133)
  lands.
- **Per-segment calibration scales and a fiber-fraction prior.** The batch carries one calibration
  polynomial per galaxy. Use the per-galaxy `Fitter` for per-segment scales until
  [#3134](https://github.com/suchethac/tengri/issues/3134) lands.
- **The MEADS sampler.** The catalog engine offers `sampler="nuts"`, `"hmc"` and `"chees"`. Use
  `sampler="chees"` until [#3135](https://github.com/suchethac/tengri/issues/3135) lands.
- **A model-centered line prior.** The batched stack takes the flat prior on line amplitudes. Use
  the flat prior until [#3138](https://github.com/suchethac/tengri/issues/3138) lands.
- **Student-t noise and a calibration floor.** The batched likelihood is Gaussian with no floor.
  Use Gaussian noise.
- **Spectral indices.** Use the per-galaxy `Fitter`.
- **Gaussian-process star formation histories.** Use a parametric history with the batch, or the
  per-galaxy `Fitter` for a Gaussian-process history.
- **Line-flux and line-ratio channels.** Remove them from the template, or use the per-galaxy
  `Fitter`.
