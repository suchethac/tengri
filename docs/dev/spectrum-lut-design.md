# Spectrum LUT — design

> Status: **partially implemented (2026-05-28)**. Phase 5 of the
> WavePrecomp rollout. Follows PR #167 (WavePrecomp + Taylor on
> `SEDModelComponent`).

## Implementation status (2026-05-28)

Landed and verified (exact-vs-LUT parity < 0.21% on low-R continuum;
gradients flow; no regressions):

- **Continuum spectrum precomp (W1).** `approx=SpectrumPrecomp()` now
  actually engages. The stellar component carries a pre-rebinned
  SSP×pixel LUT (fixed-z `SpectroscopicPrecomputation`) or re-interpolates
  the SSP cube to `wave_obs/(1+z)` at runtime (free-z — exact, avoids the
  feature-sweep error of a z-grid table). It publishes
  `stellar_spec_lnu_precomp` + `spec_eff_waves`; dust publishes per-pixel
  transmission (single + two-component — **no Taylor moment**, a pixel is
  one wavelength so `A(λ_pix)` is exact); `predict_spectrum_via_precomp`
  composes emitters × transmission × cosmology.
- **High-R auto-fallback.** `SpectrumPrecomp()` on R > 3000 spectroscopy
  warns and falls back to the exact path (`_SPECTRUM_PRECOMP_R_MAX`).
- **Emission lines on the precomp path (W2, option A below).** The
  line-publishing-backend refusal is **lifted**: `_apply_spec_precomp` now
  preserves the component's published dict, so a nebular backend's
  grid-independent `line_waves`/`line_lums` survive the LUT path.
  `predict_line_fluxes` and `pred.lines.*` are **bit-identical** to the
  exact path under `SpectrumPrecomp`.
- **Measured line ratios as data (W4).** New `LineRatioData`
  (`observation/line_ratio_data.py`), `Observation.line_ratios`,
  `SEDModel.predict_line_ratios` (single chain eval, precomp==exact), and
  a `line_ratio_constraint` adapter in the likelihood cohort
  (`build_likelihood_extras`) + fall-through. Public-API exported.

- **Window-based measurables through precomp (W3).** Fixed the
  pre-existing `predict_spectral_indices` bug (it passed a custom 2000-pt
  grid to `predict_spectrum`, which ignores it → shape mismatch). Indices
  are rest-frame quantities, so they now measure on the **attenuated
  rest-frame SED** (`predict_rest_sed`) — correct on both exact and
  precomp paths (dust sets `sed_intrinsic` to the attenuated SED in all
  cases). D4000 / Dn4000 / Balmer break / Lick EW (12+ standard) work as
  data; **multiple indices** compose into the cohort together. Added a new
  `index_type="slope"` so **UV-slope β** is a first-class index
  (`STANDARD_INDICES["uv_slope_beta"]`), matching `pred.sed.uv_slope_beta`
  and exact==precomp. **No grid augmentation needed** — measurables read
  the full rest SED / grid-independent `line_lums`, never the user's
  coarse pixel grid, so "user provides the grid, measurables always
  computed" falls out for free. Also dropped the `Observables`
  `NotImplementedError` for line_fluxes/indices: those are scalar
  measurables fed to the likelihood via the prediction dict, not
  projection fields.

Still open: joint photometry+spectroscopy under `SpectrumPrecomp` (raises
`NotImplementedError` today — needs the phot LUT family built alongside
the spec LUT). A separate abstract `Measurable` Protocol was considered
and **declined** (YAGNI): `LineFluxData` / `LineRatioData` /
`SpectralIndexData` (EW/break/slope) already form an extensible
measurables family — a new measurable is a new `index_type` or a small
data container, no protocol churn required.

## What this is

`WavePrecomp` today accelerates broadband **photometry** fits by evaluating
each parameter-dependent factor at one wavelength per filter (the
Zacharegkas+2025 effective-wavelength trick, arXiv:2506.19919). For
**spectroscopy** we still integrate the full SED on the rest-frame
wavelength grid (~6900 points for MILES, ~10k for high-R observations),
multiply by attenuation/IGM, then resample to the observed pixel grid.

The cost per likelihood call is dominated by the wave-grid traversal. A
spectrum LUT does for spectra what the filter LUT does for photometry:
collapses the most expensive parameter-independent integral once at
compile time so the JIT step only sees parameter-dependent scalars at
each output pixel.

## When the trick applies

The Zacharegkas+2025 approximation is "the parameter-dependent factor
is approximately constant across the support of the projection kernel."

| Setting              | Kernel width                   | Approximation accuracy |
|----------------------|--------------------------------|------------------------|
| Broadband photometry | filter FWHM ~ 1000 Å           | 0.5% on magnitudes     |
| Low-R spectroscopy   | pixel resolution ~ 5–50 Å      | comparable to filters  |
| Medium-R (R ~ 2000)  | resolution ~ 2–5 Å             | depends on physics — dust/AGN smooth, nebular line centers NOT |
| High-R (R > 10000)   | resolution < 1 Å               | the approximation breaks for any non-smooth feature |

The honest answer: **a per-pixel-effective-wavelength LUT works for
low-to-medium R**, breaks for high-R near emission lines. The
implementation should refuse to opt in for high-R spectra (clear error
at construction) or fall back to the exact path automatically.

## The shape

```python
@dataclass(frozen=True)
class SpectrumPrecomp:
    """Opt-in: precompute SSP × dust × IGM at spectrum pixel centers."""
    # No tuning knobs in the first cut; the pixel grid comes from
    # Observation.spectroscopy.wave_obs and z from spec/params.
```

`SEDModel.build(..., approx=SpectrumPrecomp())` would:

1. At construction time, derive `spec_eff_waves` = per-pixel effective
   wavelengths in rest-frame (= `wave_obs / (1 + z_resolved)`).
   - When `redshift` is `Fixed`, this is a static array.
   - When `redshift` is free, fall back to a ztable analogous to the
     photometry case (`WavePrecomp(z_min, z_max)`).
2. Add `spec_eff_waves` to `state.derived` so every component sees it
   alongside `filter_eff_waves`.
3. Each `SEDModelComponent.predict()` is called twice (or once with a
   joint wave grid) — once at `filter_eff_waves` for photometry and
   once at `spec_eff_waves` for spectroscopy. The framework lifts each
   into a per-component LUT keyed `<name>_phot_lnu_precomp` /
   `<name>_spec_lnu_precomp`.
4. `Observation.predict_via_precomp(state, params)` sums both LUT
   families and produces `(phot_fnu, spec_fnu)`.

## What the astronomer writes

**Nothing** — for closed-form models that already work under
`WavePrecomp(order=0)`, the same `predict()` body is called by the
framework with `wave = spec_eff_waves` instead of
`filter_eff_waves`. The body is wave-grid agnostic.

For library models (SKIRTOR, DL07): same shape — `interpolate(atlas,
p, wave)` at the smaller `spec_eff_waves` grid is just fewer points
than the full rest-frame grid.

## What needs to change

1. **`SpectrumPrecomp` dataclass** at
   `src/tengri/forward/sed_model.py` (or a sibling module). Frozen,
   no params for v0. Future: `taylor_order` mirroring `WavePrecomp`.
2. **State publication**: a new `spec_eff_waves` key in
   `DerivedBundle` and `_CANONICAL_UNITS` (units `"Å"`).
3. **Per-component LUT publishing**: extend the
   `SEDModelComponent.apply()` dispatch to produce
   `<name>_spec_lnu_precomp` LUTs in addition to the photometry LUT.
   The base class change is small — one extra branch.
4. **Observation-side consumer**:
   `observation.predict_via_precomp(state, params)` extended to fold
   the spec LUT through cosmology, IGM, and any pixel-level kernel
   (line broadening from velocity dispersion is the main one — see
   below).
5. **Compile-time validation**:
   - If `spec_eff_waves` would have neighboring pixels closer than a
     known emission-line width (depends on `line_sigma_aa`), refuse to
     opt in or warn loudly.
   - If `Spectroscopy.resolution` is set and exceeds a threshold (say
     R > 3000), refuse.
6. **Nebular lines** are the failure mode at moderate R. Two options:
   - **A.** Subtract the emission-line component from the LUT-path and
     evaluate it separately on the exact grid. Lines are a known
     publishable derived (`line_waves`, `line_lums`); the framework
     can rasterize them at observation time onto the pixel grid.
   - **B.** Refuse `SpectrumPrecomp` when a `cue`/`cloudy_grid` nebular
     backend is in the chain. Simpler v0 default; A is the follow-up.
7. **Velocity dispersion / line broadening**: today's spec path
   broadens by the prior `sigma_v` either at the SSP or pixel level.
   Under `SpectrumPrecomp`, the broadening must happen *after* the LUT
   projection (at observation time) — otherwise the LUT itself depends
   on sigma_v and the approximation collapses. The Phase 5 PR includes
   wiring this in.

## Acceptance

* Exact-vs-`SpectrumPrecomp` agreement on a representative low-R model
  (SDSS pipeline, R ~ 2000) — max relative error < 1% on the
  continuum, with line residuals separately handled via option A.
* Per-likelihood wall-clock speedup measurable on
  `bench/scripts/benchmark_forward_model.py` (extend it to include a
  spectroscopic case).
* Gradient parity through `jax.grad` of both paths on a small fit.

## Out of scope for Phase 5

* High-R spectra (refuse opt-in, document).
* Joint spec + phot LUT path optimization. The first cut runs both LUTs
  independently and sums; a future optimization can share the
  per-component primitive evaluation.
* The IFU / per-spaxel case. Same Protocol, but the spectroscopy slot
  becomes a higher-dimensional array — handled separately under the
  `SpatialModel` work (ADR-0012 + tracer-bullet from PR #159).

## File-by-file changes (sketch)

| Path                                                | Change                                                  |
|-----------------------------------------------------|---------------------------------------------------------|
| `src/tengri/forward/sed_model.py`                   | NEW: `SpectrumPrecomp` dataclass; `compile_signature` includes its config |
| `src/tengri/components/sed_model_component.py`      | Extend `_apply_precomp` to publish `<name>_spec_lnu_precomp` when `spec_eff_waves` is present |
| `src/tengri/protocols/derived_bundle.py`            | Add `spec_eff_waves` field                              |
| `src/tengri/forward/orchestrator.py`                | Add `spec_eff_waves` to `_CANONICAL_UNITS`              |
| `src/tengri/observation/predict_via_precomp.py`     | Consume `<name>_spec_lnu_precomp` and produce `spec_fnu` |
| `tests/contract/test_spectrum_lut.py`               | NEW: exact-vs-LUT agreement + gradient parity            |
| `bench/scripts/benchmark_forward_model.py`          | Extend to time spectroscopy under all approx modes      |
| `docs/dev/forward-model-architecture.md`            | Add §5.5 "Spectrum LUT" with the matrix of when it applies |

## Open design questions

1. **Joint vs split LUT.** Stellar + dust + nebular continuum LUTs at
   pixel resolution: do we publish one consolidated LUT per pixel, or
   keep the per-component LUTs and sum at observation time?
   Recommendation: per-component, mirrors photometry exactly.
2. **Nebular lines under `SpectrumPrecomp`.** Option A (lines on
   exact grid) vs Option B (refuse for line-publishing backends).
   First cut: B. Follow-up: A.
3. **`sigma_v` propagation.** Where in the pipeline does line
   broadening get applied — at SSP precompute time, at LUT publish
   time, or at observation projection time? The third option keeps
   the LUT parameter-independent, so this is recommended.

## References

* `docs/dev/forward-model-architecture.md` §5 (current WavePrecomp doc)
* Zacharegkas et al. 2025, arXiv:2506.19919
* `src/tengri/components/sed_model_component.py:_apply_precomp` (the photometry-side base implementation to mirror)
