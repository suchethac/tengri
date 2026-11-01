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
| `line_mask` | `(B, n_lines)` | 1.0 when a catalog line lands on the real grid at the galaxy's redshift; zero-width when the template fits no lines |
| `eline_resolution` | `(B,)` | Spectral resolution [dimensionless] the line profiles use, per galaxy |
| `chol_data` | `(B, p + 1, n_max)` | Bands of each galaxy's lower Cholesky factor of its banded pixel covariance, factored on the host with that galaxy's `pix_mask`; masked and padded rows are identity rows; `(B, 0, n_max)` placeholder without a covariance |
| `segment` | `(B, n_max)` | Calibration segment id of each pixel, padded with the last real id; zeros when `spec.n_segments == 0` |
| `cal_scale_center`, `cal_scale_sigma` | `(B,)` | `log10_center` and `sigma_dex` of the galaxy's log-normal prior on the mean calibration multiplier; 0 and 1 when it has none |

The batch carries no line-flux leaves: the line-flux and line-ratio channels are
refused (see the refusal table).

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
| A template `Spectroscopy.covariance` | The batched path reads each galaxy's own covariance; a template covariance would be ignored | Build the template without one and give every `GalaxySpectrum` its `covariance=BandedMatrix` (carried by the `chol_data` leaf) |
| `eline_mode='marginalized'` with `eline_prior_type='cloudy'` | The Cloudy-prior line marginal is not evaluated in batch | Use `eline_prior_type='flat'`, or `eline_mode='fitted'` or `'off'` |
| `eline_broad=True` with lines | Broad-line profiles are not evaluated in batch | Use `eline_broad=False` |
| `line_fluxes`, `line_ratios`, `spectral_indices`, `lines` | Not evaluated in batch | Remove them from the template |
| Student-t or non-zero calibration-floor noise | The batched likelihood is Gaussian with no floor | Use diagonal Gaussian noise with a zero floor |

Two refusals are raised from `inference/batched_spectro.py`, not from the table:

| Refused setting | Exception | What lifts it |
|---|---|---|
| Stochastic (GP-field) SFH | `NotImplementedError` | Use a parametric SFH |
| `calibration_marginalize=True` with `calibration_order > 0` | `ConfigError` | Set `calibration_order=0`, or set `calibration_marginalize=False` and fit the coefficients |
| `calibration_marginalize=True` with `eline_mode='fitted'` | `NotImplementedError` | Use `eline_mode='marginalized'` for the joint marginal, or set `calibration_marginalize=False` |

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

With `eline_mode='marginalized'` (flat prior) the line amplitudes are integrated out
with the same primitive. The line design is `G_eff` from `batched_eline_design`, with
a column mask for lines that land outside the real grid. The line amplitudes are
rescaled by the prior width, `a = eline_prior_sigma * a'`, so the prior is the unit
normal. The marginal is invariant under this change of variables, and the prior
precision `1/eline_prior_sigma**2` is never formed, since it overflows float32 at
F_nu scale. The reparametrization is one shared helper, `unit_line_prior` in
`observation/eline_marginalization.py`, that the batched path and both per-galaxy classes call. The joint with calibration uses the design `[spec_fnu * T(x) | G_eff * eline_prior_sigma]`
and the block prior `blockdiag(I / cal_prior_sigma**2, I)`. The per-galaxy classes
`ELineMarginalizedLikelihood` and `CalibrationELineMarginalizedLikelihood` evaluate
the same joint, and the two paths agree at rtol 1e-10 for one galaxy.

The line marginal drops `-1/2 N ln 2pi - sum ln sigma`, as the per-galaxy line marginal
does, so the joint with calibration on the marginalized path also drops that constant.

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

- 2026-11-01, #3133: banded pixel covariance on the batched spectroscopic path.
  `GalaxySpectrum(covariance=BandedMatrix)` gives a galaxy a banded covariance; every
  galaxy that carries one uses the same offsets, or `build_spectro_batches` raises naming
  both offset tuples. The builder factors `M C M + (I - M)` per galaxy on the host
  (`M` its good-pixel mask) into the `chol_data` leaf, with identity padded rows, and
  `SpectroBatchSpec.cov_offsets` holds the factor's static offsets. The Gaussian term,
  the calibration marginal and the line marginals whiten by the factor through
  `marginalize_linear_nuisance(chol_lower=...)` or `banded_forward_substitution`, so a
  masked or padded pixel contributes nothing. The Gaussian term drops the normalization
  constant, as the diagonal term and the per-galaxy `MultivariateGaussianLikelihood`
  do. A template's own `Spectroscopy.covariance` is refused.
  `tests/contract/test_batched_banded_covariance.py` pins the batched value against the
  per-galaxy dense covariance with masked pixels deleted, the marginals against the
  per-galaxy classes, padding at `rtol=1e-12`, one compile across covariances, the
  offsets refusal, and the float32 value and gradient at log10 M = 4.0.
- 2026-10-31, #3117: analytic emission-line marginalization on the batched spectroscopic
  path, and the exact joint of calibration and lines on both paths. `eline_mode="marginalized"`
  with a flat prior is accepted, alone or with `calibration_marginalize=True`. The
  per-galaxy `CalibrationELineMarginalizedLikelihood` is the exact joint, replacing the
  sequential point-estimate composition (#2354). `marginalize_emission_lines` is a wrapper
  over the primitive, and the value is the exact marginal: the prior penalty enters once.
  `tests/contract/test_batched_eline_marginal.py` pins the batched value against the
  per-galaxy classes, a masked line against the run with that column deleted, every line
  masked against the lines-off likelihood, one compile across masks, padding, and the
  float32 value and gradient at log10 M = 4.0. The slow-tier comparison of batched and
  per-galaxy NUTS medians is in `tests/inference/test_batched_spectro_catalog.py`.
- 2026-11-01, #3134: per-segment calibration scales, the scale-free coefficient prior and the
  fiber-fraction prior on both paths. `Spectroscopy(calibration_segments=, cal_prior=,
  cal_scale_prior=, cal_scale_window=)` and `GalaxySpectrum(segment_ids=, cal_scale_prior=)` feed
  the leaves above; `SpectroBatchSpec.n_segments` is static and part of the bucket key, so
  galaxies with different segment counts land in different buckets. The per-galaxy
  `CalibrationMarginalizedLikelihood`, the joint class and the batched marginal all call
  `marginalize_calibration_segments`, and on the segment route the banded pixel covariance factor
  enters that same marginal (`chol_data=`, `chol_offsets=`). The mean multiplier over the window is carried as the
  latent `cal_log_q` (`ln q`): per galaxy a declared observation parameter with the Gaussian
  prior from `cal_scale_prior`; batched, a latent with a unit-normal template prior whose
  standardized prior term `batched_neg_log_posterior` replaces by each galaxy's own
  log-normal. The coefficient block is conditioned on `q` through an orthonormal null-space
  basis of the window vector, and the value includes the coefficient prior's own marginal of
  `ln q` (`calibration_q_log_prior`). `tests/contract/test_calibration_segments_scale_prior.py`
  pins the primitive layer and the quadrature over `ln q`;
  `tests/contract/test_calibration_segments_batched.py` pins batched against per-galaxy,
  padding, one compile and float32.
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
- 2026-11-01, #3138: `eline_prior_type="model"` on the per-galaxy line-only marginalized
  likelihood. Each line has a Gaussian prior centered on the model's attenuated line
  luminosity, `mu = f_ap L lambda_obs^2 / (4 pi d_L^2 c)`, with width `(10^dex - 1) mu`.
  The mapping is checked against the lines-on minus lines-off spectrum over the H-beta
  window (ratio 0.991). The prior mean carries the Milky Way foreground screen at each
  line's observed wavelength on both paths; the intergalactic transmission is left out,
  since every catalog line lies above 1216 Angstrom, where it transmits exactly one.
  The batched path evaluates the same prior through the same
  helpers (`model_line_prior`, `unit_model_line_prior`), with the attenuated line
  luminosities returned by `predict_batched_observables` and the catalog-to-model line
  map fixed at build time; the calibration joint takes the block prior
  `blockdiag(I / sigma_cal^2, unit)` with prior mean `[0, mu / w]` on both paths.
  `eline_aperture_frac` is a registered observation parameter (`Fixed(1.0)`, freed with
  `eline_aperture_frac=Uniform(...)` or `FREE` at build, round-tripped by `to_groups`).
  The per-galaxy fit reads the line luminosities from the same forward that paints the
  spectrum. A masked line column leaves no trace in the marginal even with a nonzero
  prior mean.
