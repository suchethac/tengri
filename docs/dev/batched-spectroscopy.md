# The batched spectroscopic stack

This page is for contributors. It describes how a catalog of spectra is evaluated
through one compiled function, what the traced leaves are, which template settings
the batched path refuses, and how the calibration marginalization is shared with the
per-galaxy path.

## What it is

A catalog fit evaluates many galaxies with the same forward model. Their pixel
grids, pixel counts, resolution operators and redshifts differ, so the observation
enters the compiled function as an argument rather than as a closed-over constant.

- `observation/batched.py` builds one **`SpectroBatch`** per bucket: a pytree whose
  leaves all carry a leading batch axis `B`. The leaves are traced.
- Galaxies are grouped into buckets by a **`SpectroBatchSpec`**, a frozen static key.
  Every galaxy in a bucket shares that spec, so the bucket compiles once.
- `forward/batched_spectrum.py` evaluates one galaxy's spectrum from its slice of the
  batch. `inference/batched_spectro.py` adds the Gaussian data term, the standardized
  negative log-posterior, the vmapped MAP and the catalog MCMC driver.

## The leaves

Every entry of `_BATCH_FIELDS` in `observation/batched.py`, in order:

| Leaf | Shape | How it is built |
|---|---|---|
| `wave` | `(B, n_max)` | Observed grid [Angstrom], padded past the real pixels |
| `pix_mask` | `(B, n_max)` | 1.0 on a good real pixel; 0.0 on padding, masked pixels and `ivar <= 0` |
| `flux` | `(B, n_max)` | F_nu [erg/s/cm^2/Hz]; 0.0 on excluded pixels |
| `sigma` | `(B, n_max)` | 1-sigma error; 1.0 on excluded pixels (keeps the whitening finite) |
| `r_data` | `(B, K, n_max)` | Banded resolution rows, `K` offsets; `K = 0` placeholder for a Gaussian `R` |
| `resolution_pp` | `(B, n_max)` | Per-pixel `R`; `(B, 0)` placeholder for a banded operator |
| `z` | `(B,)` | Redshift of each galaxy |
| `cal_lo`, `cal_hi` | `(B,)` | Smallest and largest real wavelength [Angstrom]; padded pixels lie outside `[-1, 1]` in the calibration coordinate but carry zero weight |
| `phot_flux`, `phot_err` | `(B, n_filt)` | Photometry; zero-width when the bucket has none |
| `phot_presence` | `(B, n_filt)` | 1.0 where a filter has a finite flux and a positive error |
| `sfh_bin_edges_gyr` | `(B, n_sfh_edges)` | Per-galaxy SFH bin edges [Gyr]; zero-width when the template ladder is used |

The batch carries no line-flux, line-mask or emission-line resolution leaves: the
emission-line channels are refused (see the refusal table).

## The key fields of `SpectroBatchSpec`

| Field | Meaning |
|---|---|
| `n_max` | Padded pixel count of every galaxy in the bucket |
| `offsets` | Diagonal offsets of the banded operator; `None` for a Gaussian `R` |
| `grid_kind` | `"banded"`, `"log_uniform"` or `"nonuniform"`; decided from concrete arrays at build time, so a traced grid takes the general path |
| `lsf_n_bins` | Bins of the spectral LSF approximation |
| `has_phot`, `n_filt` | Whether the bucket carries photometry, and how many filters |
| `conserving` | Flux-conserving pixel integral, from the template's `resample` mode |
| `lsf_pad_pixels` | Static FFT padding per side, from the concrete real grids |
| `n_sfh_edges` | Length of the per-galaxy SFH ladder; `0` when the template's ladder is used |

Fields that change array shapes or control flow are in the key, so a change to any
of them is a new bucket and a new compile.

## The refusal table

`check_template_model` in `forward/batched_spectrum.py` refuses a template before any
batch is evaluated. Each refusal below is raised as a `ValueError` unless noted.

| Refused setting | Reason | What lifts it |
|---|---|---|
| No spectroscopy configured | There is nothing to evaluate | Configure `Spectroscopy` |
| Fixed redshift | The batch's `obs.z` must be the only redshift | Build with a free redshift |
| `catalog_z_range` | A per-galaxy redshift window in `WavePrecomp` | Build with a free redshift and without `catalog_z_range` |
| `SpectrumPrecomp` spectrum LUT | The LUT bakes the observation grid | Build with `spectrum_precomp` off |
| Spectral covariance | Correlated noise; the batched likelihood is diagonal | Use diagonal noise |
| `eline_mode != 'off'` | Emission-line fitting is not evaluated in batch | Use `eline_mode='off'` |
| `line_fluxes`, `line_ratios`, `spectral_indices`, `lines` | Not evaluated in batch | Remove them from the template |
| Student-t or non-zero calibration-floor noise | The batched likelihood is Gaussian with no floor | Use diagonal Gaussian noise with a zero floor |

Two refusals are raised from `inference/batched_spectro.py`, not from the table:

| Refused setting | Exception | What lifts it |
|---|---|---|
| Stochastic (GP-field) SFH | `NotImplementedError` | Use a parametric SFH |
| `calibration_marginalize=True` with `calibration_order > 0` | `ConfigError` | Set `calibration_order=0`, or set `calibration_marginalize=False` and fit the coefficients |

## One marginalization primitive

`observation/linear_marginal.py::marginalize_linear_nuisance` is the only
marginalization math on either path. The data model is a known offset plus a linear
combination of nuisance coefficients, with a Gaussian prior on the coefficients:

- **Masked pixels** drop out of every term: the effective count, the `sum ln sigma`,
  the residual and the design. Masked pixels are selected, not floored.
- **Masked columns** are fixed out of the model by an identity block. A masked
  coefficient contributes exactly nothing, provided its prior-mean entries are zero.
- **The scale** `s` rescales the coefficients so the whitened design stays
  representable in float32 when the raw design is at flux scale. The value does not
  depend on `s` in exact arithmetic.
- **The flat-prior convention**: with no prior precision the improper constant is
  fixed by keeping the Jacobian `+ n_c ln s` of the change of variables, so the
  constant is the same in every unit system.

Two adapters sit on top of the primitive: `inference/likelihoods/marginalized.py`
(per galaxy) and `inference/batched_spectro.py` (batched). The calibration basis is
`observation/calibration.py::calibration_basis`, a Chebyshev series on the grid's
extremes.

With `calibration_marginalize=True` the batched spectral term is

`marginalize_linear_nuisance(data=obs.flux, model0=spec_fnu, design=spec_fnu[:, None] * T(x), sigma=obs.sigma, pix_mask=obs.pix_mask, prior_precision=I / cal_prior_sigma**2).log_like`

with `T(x)` built on `cal_lo`, `cal_hi`. The photometric term is unchanged.

**Normalization.** The plain batched spectral term drops
`-1/2 N ln 2pi - sum ln sigma` over the real pixels, as `diag_gaussian_log_prob` does.
The marginal keeps it, as `CalibrationMarginalizedLikelihood` does on the per-galaxy
path. The two settings therefore differ by that constant and by nothing else.

## The homogeneity rule

Every galaxy in a bucket must share one `SpectroBatchSpec`, and padding must not
change any real pixel. `build_spectro_batches` checks the second part and raises
`ValueError`, naming the galaxies whose padding would change real pixels, with the
hint to use `quantum=None`. A new static field belongs in `SpectroBatchSpec`, not in
a leaf, if it changes a shape or a branch.

## Record

- 2026-10-31, #3116: calibration marginalization on the batched spectroscopic path.
  `batched_log_likelihood`, `batched_neg_log_posterior`, `fit_spectra_map_vmap` and
  `fit_spectra_catalog_mcmc` take `calibration_marginalize`, `cal_n_poly` and
  `cal_prior_sigma`, which the batched spectral term passes to
  `marginalize_linear_nuisance`. The primitive itself is the part-A commit this lane
  merged. `tests/contract/test_batched_cal_marginal.py` pins the single-galaxy value
  and gradient against `CalibrationMarginalizedLikelihood`, padding and compile counts,
  the `ConfigError`, and the MAP tilt absorption. Open at the time of writing: float32
  mode (`jax.enable_x64(False)`) returns NaN for this fixture's flux scale on the
  batched forward, for the plain Gaussian term as well as the marginal, so the float32
  test is not yet green.
