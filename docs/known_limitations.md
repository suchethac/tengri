# Known limitations

tengri is research code under active development (v0.1.0). The forward model
and inference pipeline work and are covered by 2000+ tests, but the project is
pre-publication and several capabilities are partial. This page collects the
limitations a user is most likely to hit — better to read it before a fit
fails than after.

## Verification status

No physics component has yet completed a full external cross-validation
(against FSPS, Prospector, or CIGALE). The reproduction notebooks under
`docs/reproduction/` compare specific components against established codes, but
the formal verification protocol (`docs/dev/verification-protocol.md`) lists
every major block as still in progress. **Do not treat fitted parameters as
publication-grade until the component you rely on has been cross-validated.**

## Approximations

- **`WavePrecomp` accelerates the photometry channel.** It precomputes the
  SSP × filter integrals for a large speedup, with free redshift handled via an
  interpolation table. The stellar photometry is ~0.4 % accurate, **but dust is
  re-applied as a first-order Taylor projection across each filter
  (suchethac/tengri#617)**. That linear-in-λ model is accurate where the
  attenuation curve is smooth across a band (optical / IR) but biases bands
  sampling the **rest-UV**, where the curve is steep and extrapolated — by an
  order of magnitude for far-UV bands at moderate/high redshift (e.g. at
  z ≈ 2 with τ ≈ 0.5: SDSS *u/g* ~5–10 % high, GALEX FUV off by >10×; with zero
  dust the LUT is exact). Such configurations (`WavePrecomp` + a non-trivial
  dust screen + a rest-UV band) emit a build-time `UserWarning`. For unbiased
  blue-band photometry use `approx=None`, or validate against it. The full fix
  (a per-band exact-integral fallback in the steep-attenuation regime) is
  tracked separately.
- **`SpectrumPrecomp` accelerates the spectroscopy channel.** It precomputes a
  per-pixel effective-wavelength continuum lookup table. It agrees with the
  exact path to machine precision with zero dust; with dust it carries the same
  *class* of intra-element attenuation error as `WavePrecomp` but **far milder**,
  because a pixel is narrow (tens of Å) rather than a full filter — typically
  ≲1 % on the continuum, growing to ~10–15 % only in the deep rest-UV at high
  optical depth. At high spectral resolution it auto-falls-back to the exact
  wave-grid path with a warning. **Component coverage:** stellar, dust IR, radio,
  X-ray, **AGN**, and non-baked **nebular** (``Cue`` / ``CloudyGrid``) are all
  included in the spectrum LUT. Nebular emission is reddened by the young-limit
  screen (birth cloud + diffuse), matching the exact path. (Earlier the AGN was
  dropped, and Cue/CloudyGrid nebular emission lines were dropped from
  ``predict_spectrum`` then over-counted ~2.3× by diffuse-only attenuation — both
  fixed.) ``BakedIn`` nebular (baked into the SSP) is carried by the stellar LUT.
- **Joint photometry + spectroscopy is supported under precompute.** On a joint
  observation, either precompute opt-in (`approx=WavePrecomp()` or
  `approx=SpectrumPrecomp()`) builds **both** LUT families; the forward pass
  projects photometry and spectroscopy together inside one fused, cached JIT
  kernel. For independent per-channel tuning, pass a composite tuple —
  `approx=(WavePrecomp(n_z=200), SpectrumPrecomp())` — which accelerates both
  channels with each LUT configured separately (suchethac/tengri#610). Velocity
  dispersion / LSF are not applied on the per-pixel continuum LUT
  (`SpectrumPrecomp`'s documented low-to-medium-R domain) — use `approx=None` if
  you need the exact LSF-convolved spectrum.

## Inference backends

`tengri.list_inference_methods()` tags each backend `primary` or
`experimental`. Several experimental backends carry explicit `[POOR MIXING]` or
`[UNSTABLE]` flags in their description (e.g. `mcmc_ghmc`, `mcmc_mclmc`,
`pathfinder`, `native_vi_*`). Stick to `primary` backends for science, and if
you do try an experimental one, read its flag first. See
[Choosing an inference method](method_selection).

## Memory

NUTS/VI warmup can peak far above the resident forward model — 20+ GB on a
`dense_basis` D≈8 model with `dense_mass_matrix=True`. Practical rules:

- Run **one** NUTS/VI fit per Python process.
- On D ≥ 8, use `dense_mass_matrix=False` or `mcmc_hmc`.
- `WavePrecomp` (photometry) keeps fits in the few-GB range.

Full operational detail: `docs/dev/notebook_orchestration_oom.md`.

## Physics caveats

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
