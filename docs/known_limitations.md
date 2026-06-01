# Known limitations

tengri is research code under active development (v0.1.0). The forward model
and inference pipeline work and are covered by 2000+ tests, but the project is
pre-publication and several capabilities are partial. This page collects the
limitations a user is most likely to hit. It is deliberately honest — prefer
reading it before a fit fails to reading it afterwards.

## Verification status

No physics component has yet completed a full external cross-validation
(against FSPS, Prospector, or CIGALE). The reproduction notebooks under
`docs/reproduction/` compare specific components against established codes, but
the formal verification protocol (`docs/dev/verification-protocol.md`) lists
every major block as still in progress. **Do not treat fitted parameters as
publication-grade until the component you rely on has been cross-validated.**

## Approximations

- **`WavePrecomp` is photometry-only.** It precomputes the SSP × filter
  integrals for a large speedup on photometric fits, with free redshift handled
  via an interpolation table. It does not apply to spectroscopy.
- **`SpectrumPrecomp` is spectroscopy-only.** It precomputes a per-pixel
  effective-wavelength continuum lookup table for spectroscopic fits. It runs
  (it does not raise) and agrees with the exact path to machine precision on
  the stellar continuum; the two-component birth-cloud dust LUT carries a known
  ~1% residual. At high spectral resolution it auto-falls-back to the exact
  wave-grid path with a warning.
- **Neither precompute path applies to joint (photometry + spectroscopy)
  fits.** A model accepts a single `approx=` object; a joint model built with
  `SpectrumPrecomp()` raises `NotImplementedError` (the photometric LUT family
  is not yet built alongside the spectrum LUT). Build a joint model with
  `approx=None` (exact) for now.

## Inference backends

`tengri.list_inference_methods()` tags each backend `primary` or
`experimental`. Several experimental backends carry explicit `[POOR MIXING]` or
`[UNSTABLE]` flags in their description (e.g. `mcmc_ghmc`, `mcmc_mclmc`,
`pathfinder`, `native_vi_*`). Stick to `primary` backends for science and read
the flag before reaching for an experimental one. See
[Choosing an inference method](method_selection).

## Memory

NUTS/VI warmup can peak far above the resident forward model — 20+ GB on a
`dense_basis` D≈8 model with `dense_mass_matrix=True`. Practical rules:

- Run **one** NUTS/VI fit per Python process.
- On D ≥ 8, use `dense_mass_matrix=False` or `mcmc_hmc`.
- `WavePrecomp` (photometry) keeps fits in the few-GB range.

Full operational detail: `docs/dev/notebook_orchestration_oom.md`.

## Physics caveats worth knowing

- **AGN torus:** the toy torus models in `agn/torus.py` are not for science —
  use `agn_model="skirtor"` (SKIRTOR; Stalevski et al. 2016).
- **Dust IR emission templates** load from `data/`; the analytic fallbacks are
  not suitable for science.
- **Nebular backends:** the `Cue` and `CloudyGrid` backends require a
  bare-stellar SSP grid; `BakedIn` (nebular baked into the SSP at fixed `logU`)
  cannot vary `neb_logU` / `neb_fesc`. Four of the six `tengri.recipes` need a
  bare-stellar grid — see [Recipes](recipes).
- **Metallicity** is `log10(Z)` absolute on the SSP grid; the user-facing
  `neb_logZ_gas` is `Z/Z_sun`.

## Reporting

Found something not listed here? Open an issue at
<https://github.com/suchethac/tengri/issues>.
