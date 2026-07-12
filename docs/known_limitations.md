# Known limitations

Tengri is research code under active development (v0.1.0). The forward model
and inference pipeline work and are covered by 2000+ tests, but several
capabilities are partial. This page collects the limitations you are most
likely to hit.

## Verification status

No physics component has completed full external cross-validation
against FSPS, Prospector, or CIGALE. The reproduction notebooks under
`docs/reproduction/` compare specific components against established codes, but
the formal verification protocol (`docs/dev/verification-protocol.md`) lists
every major block as still in progress. **Do not treat fitted parameters as
publication-grade until your component has been cross-validated.**

## Approximations

- **`WavePrecomp` accelerates the photometry channel.** It precomputes the
  SSP × filter integrals, with free redshift via interpolation. Each filter is
  split into `n_subbands` sub-bands of equal filter mass (default 5), and the
  multiplicative screens — dust attenuation and IGM transmission — are
  **evaluated** at each sub-band's quadrature node, the template's own
  flux-weighted centroid there. Measured against the exact path across
  GALEX→WISE at z ≤ 1.5 with τ_diff = 0.7 / τ_bc = 1.0, the worst band agrees to
  ≲0.5%, and the optical/NIR bands to ≲0.01%. Accuracy improves as 1/K²; raise
  `WavePrecomp(n_subbands=8)` for the deep rest-UV. Emission that the quadrature
  cannot reach — nebular lines, AGN, dust IR — is still projected at the filter's
  effective wavelength.
- **`WavePrecomp(n_subbands=0)` restores the old first-order Taylor projection**
  (suchethac/tengri#617), which extrapolates the screen from each filter's
  effective wavelength instead of evaluating it. That linear-in-λ model is
  accurate for smooth attenuation curves (optical/IR) but biases rest-UV bands
  where the curve is steep: at z ≈ 2 with τ ≈ 0.5, SDSS *u/g* run 5–10% high and
  GALEX FUV >10× high. A build-time `UserWarning` flags such configurations. It
  is retained only for reproducing pre-#1122 results.
- **`SpectrumPrecomp` accelerates the spectroscopy channel.** It precomputes a
  per-pixel effective-wavelength continuum LUT. It agrees with the exact path
  to machine precision with zero dust. With dust, it carries the same error
  class as `WavePrecomp` but far milder (pixel-wide vs filter-wide): typically
  ≲1% on the continuum, growing to ~10–15% only in the deep rest-UV at high
  optical depth. High spectral resolution auto-falls-back to the exact
  wave-grid path. **Component coverage:** stellar, dust IR, radio, X-ray, AGN,
  and non-baked nebular (Cue/CloudyGrid) are included in the spectrum LUT.
  Nebular emission is reddened by the young-limit screen, matching the exact
  path. BakedIn nebular is carried by the stellar LUT.
- **Joint photometry + spectroscopy is supported under precompute.** Either
  precompute opt-in builds both LUT families; the forward pass projects both
  channels in one fused, cached JIT kernel. For independent per-channel tuning,
  pass `approx=(WavePrecomp(n_z=200), SpectrumPrecomp())` (suchethac/tengri#610).
  Velocity dispersion / LSF are not applied on the per-pixel continuum LUT
  (`SpectrumPrecomp`'s low-to-medium-R domain). Use `approx=None` for the exact
  LSF-convolved spectrum.

## Inference backends

Each backend is tagged `primary` or `experimental`. Experimental backends
carry explicit `[POOR MIXING]` or `[UNSTABLE]` flags (e.g. `mcmc_ghmc`,
`mcmc_mclmc`, `pathfinder`, `native_vi_*`). Use `primary` backends for science.
See [Choosing an inference method](method_selection).

## Memory

NUTS/VI warmup peaks far above the resident forward model — 20+ GB on
`dense_basis` D≈8 with `dense_mass_matrix=True`. Rules:

- Run **one** NUTS/VI fit per Python process.
- On D ≥ 8, use `dense_mass_matrix=False` or `mcmc_hmc`.
- `WavePrecomp` keeps photometry fits in the few-GB range.

Full detail: `docs/dev/notebook_orchestration_oom.md`.

## Physics caveats

- **AGN torus:** `agn/torus.py` toy models are not for science. Use
  `agn_model="skirtor"` (Stalevski et al. 2016).
- **Dust IR emission templates** must be loaded from `data/`; analytic
  fallbacks are not suitable for science.
- **Nebular backends:** `Cue` and `CloudyGrid` require bare-stellar SSP grids;
  `BakedIn` cannot vary `neb_logU` / `neb_fesc`. Four of six `tengri.recipes`
  need bare-stellar grids.
- **Metallicity** is `log10(Z)` absolute on the SSP grid; user-facing
  `neb_logZ_gas` is `log10(Z/Z_sun)`.

## Reporting

Found something not listed here? Open an issue at
<https://github.com/suchethac/tengri/issues>.
